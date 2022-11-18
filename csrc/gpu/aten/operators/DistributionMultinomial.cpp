#include <ATen/ATen.h>

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/SYCLGroupAlgorithm.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename scalar_t, typename item_t>
inline void renormRowsL1(
    item_t& item,
    scalar_t* dist,
    long rows,
    long cols,
    unsigned char* my_smem) {
  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  scalar_t* smem = reinterpret_cast<scalar_t*>(my_smem);
  scalar_t zero = static_cast<scalar_t>(0);
  scalar_t val;
  for (int64_t row = group_idx; row < rows; row += group_range) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t col = thread_idx; col < cols; col += thread_range) {
      val = dist[row * cols + col];
      sum = sum + val;
    }

    sum = GroupReduceSum(item, sum, smem);
    if (thread_idx == 0) {
      smem[0] = sum;
    }
    item.barrier(dpcpp_local_fence);

    sum = smem[0];
    if (sum > zero) {
      for (int64_t col = thread_idx; col < cols; col += thread_range) {
        dist[row * cols + col] = dist[row * cols + col] / sum;
      }
    }
  }
}

inline void renormRows(Tensor& t) {
  TORCH_CHECK(t.dim() == 2);
  int64_t rows = t.size(0);
  int64_t cols = t.size(1);

  int group_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  int num_groups = (rows + group_size - 1) / group_size;
  int hw_max_groups = dpcppMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;

  auto& sycl_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(t.scalar_type(), "renormRows", [&] {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto slm = dpcpp_local_acc_t<scalar_t>(
          (group_size / 8) * sizeof(scalar_t),
          cgh); // We use the smallest subgroup size to ensure enough space
      auto t_ptr = t.data_ptr<scalar_t>();
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        renormRowsL1<scalar_t>(
            item, t_ptr, rows, cols, (unsigned char*)slm.get_pointer().get());
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  });
}

template <typename scalar_t>
inline int binarySearchForMultinomial(
    scalar_t* cumdist,
    scalar_t* dist,
    int size,
    scalar_t val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    scalar_t midVal = cumdist[mid];
    if (Numerics<scalar_t>::lt(midVal, val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  while (start >= 1 && dist[start] == 0)
    start--;

  return start;
}

template <typename scalar_t, typename item_t>
inline void sampleMultinomialWithReplacement(
    item_t& item,
    PhiloxState philox_args,
    int totalSamples,
    int64_t* dest,
    int64_t distributions,
    int categories,
    scalar_t* normDistPrefixSum,
    scalar_t* normDist) {
  auto thread_idx = item.get_local_id(1);
  auto thread_range = item.get_local_range(1);
  auto group_idx_x = item.get_group(1);
  auto group_idx_y = item.get_group(0);
  auto group_range_x = item.get_group_range(1);
  auto group_range_y = item.get_group_range(0);

  // At the moment, each subgroup computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  auto seeds = philox_unpack(philox_args);

  // global index formula for 2D grid of 1D group
  int idx = group_idx_y * group_range_x * thread_range +
      group_idx_x * thread_range + thread_idx;

  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = group_idx_y; curDist < distributions;
       curDist += group_range_y) {
    for (int sample = group_idx_x * thread_range + thread_idx;
         sample < totalSamples;
         sample += thread_range * group_range_x) {
      // we are losing 3 out of 4 generated numbers but it's ok
      // this kernel is not very efficient anyway
      auto rand = rand_uniform4(&state);
      scalar_t r = static_cast<scalar_t>(rand.x);

      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          normDist + curDist * categories,
          categories,
          r);

      dest[curDist * totalSamples + sample] = choice;
    }
  }
}

void multinomial_with_replacement_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    c10::optional<Generator> generator) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      generator, xpu::dpcpp::detail::getDefaultDPCPPGenerator());

  int inputSize = self.dim();
  int64_t numDist = inputSize == 1 ? 1 : self.size(0);
  int numCategories = inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;

  result.resize_({numDist, n_sample});

  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      self_v.scalar_type(), "multinomial_kernel", [&] {
        using accscalar_t = acc_type<scalar_t>;

        Tensor origDist = at::empty_like(self_v);
        origDist.copy_(self_v);

        Tensor normDist = at::empty_like(self_v);

        Tensor prefixSum = at::empty_like(self_v);

        // Renorm along rows
        normDist.copy_(origDist);
        renormRows(normDist);

        // Prefix sum along rows
        at::cumsum_out(prefixSum, normDist, 1);

        int group_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
        int group_range_y = numDist;
        int group_range_x = (n_sample - 1) / group_size + 1;

        std::pair<uint64_t, uint64_t> rng_engine_inputs_;
        {
          // See Note [Acquire lock when using random generators]
          std::lock_guard<std::mutex> lock(gen->mutex_);
          auto offset = ((numDist - 1) / group_range_y + 1) * 4;
          rng_engine_inputs_ = gen->philox_engine_inputs(offset);
        }
        auto rng_engine_inputs = PhiloxState(
            std::get<0>(rng_engine_inputs_), std::get<1>(rng_engine_inputs_));
        // Sample with replacement

        auto cgf = DPCPP_Q_CGF(cgh) {
          auto result_ptr = result.data_ptr<int64_t>();
          auto prefixSum_ptr = prefixSum.data_ptr<scalar_t>();
          auto normDist_ptr = normDist.data_ptr<scalar_t>();
          auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
            sampleMultinomialWithReplacement(
                item,
                rng_engine_inputs,
                n_sample,
                result_ptr,
                numDist,
                numCategories,
                prefixSum_ptr,
                normDist_ptr);
          };
          cgh.parallel_for(
              sycl::nd_range<2>(
                  sycl::range<2>(group_range_y, group_range_x * group_size),
                  sycl::range<2>(1, group_size)),
              kfn);
        };
        DPCPP_Q_SUBMIT(sycl_queue, cgf);
      });

  if (inputSize == 1) {
    result.resize_({n_sample});
  }
}

} // namespace impl

// TODO: FLT_MANT_DIG is 24?
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (24);

Tensor& multinomial_out(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen,
    Tensor& result) {
  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ",
      result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(
      with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }

  // Fast-path for no replacement.
  // Reference:
  // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  if (!with_replacement) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    if (self.dim() == 1) {
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, gen);
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on XPU side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  impl::multinomial_with_replacement_kernel_impl(result, self, n_sample, gen);
  return result;
}

Tensor multinomial(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen) {
  Tensor result = at::empty({0}, self.options().dtype(kLong));
  multinomial_out(self, n_sample, with_replacement, gen, result);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
