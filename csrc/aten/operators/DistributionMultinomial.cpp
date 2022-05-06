#include <ATen/ATen.h>

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "Distributions.h"
#include "Random.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
int binary_search_for_multinomial(
    scalar_t* cumdist,
    scalar_t* dist,
    int size,
    scalar_t val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  // assert(THDPCPPNumerics<scalar_t>::gt(cumdist[size - 1], 0));

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

  while (start >= 1 && Numerics<scalar_t>::eq(dist[start], 0))
    start--;

  return start;
}

template <typename scalar_t>
void sample_multinomial_with_replacement(
    std::pair<uint64_t, uint64_t> seeds,
    int num_samples,
    int64_t distributions,
    int categories,
    at::Tensor& result,
    at::Tensor& norm_dist_prefix_sum,
    at::Tensor& norm_dist) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  size_t work_group_size = dpcppMaxWorkGroupSize(dev_id);
  // Each work item in a work group will generate a sample from one
  // distribution concurrently
  work_group_size = std::min(work_group_size, (size_t)num_samples);
  DPCPP::range<2> global_range(distributions, work_group_size),
      local_range(1, work_group_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto result_data = result.data_ptr<long>();
    auto norm_dist_prefix_sum_data = norm_dist_prefix_sum.data_ptr<scalar_t>();
    auto norm_dist_data = norm_dist.data_ptr<scalar_t>();
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      size_t dist_id = item_id.get_group(0);
      size_t sample_id = item_id.get_local_id(1);
      size_t global_linear_id = item_id.get_global_linear_id();
      long* result_ptr = result_data;
      scalar_t* norm_dist_prefix_sum_ptr = norm_dist_prefix_sum_data;
      scalar_t* norm_dist_ptr = norm_dist_data;

      RandomState<Philox4_32_10> state(
          seeds.first, global_linear_id, seeds.second);

      for (int sample = sample_id; sample < num_samples;
           sample += work_group_size) {
        float rand = state.uniform<float>();
        scalar_t r = ScalarConvert<float, scalar_t>::to(rand);

        // Find the bucket that a uniform sample lies in
        int choice = binary_search_for_multinomial<scalar_t>(
            norm_dist_prefix_sum_ptr + dist_id * categories,
            norm_dist_ptr + dist_id * categories,
            categories,
            r);

        // Torch indices are 1-based
        result_ptr[dist_id * num_samples + sample] = choice;
      }
    };

    cgh.parallel_for(DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
void sample_multinomial_without_replacement(
    std::pair<uint64_t, uint64_t> seeds,
    int num_samples,
    int sample,
    int64_t distributions,
    int categories,
    at::Tensor& result,
    at::Tensor& norm_dist_prefix_sum,
    at::Tensor& norm_dist) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  size_t work_group_size = dpcppMaxWorkGroupSize(dev_id);
  // Each work item in a work group will generate a sample from one
  // distribution concurrently
  work_group_size = std::min(work_group_size, (size_t)num_samples);
  DPCPP::range<1> range(distributions);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto result_data = result.data_ptr<long>();
    auto norm_dist_prefix_sum_data = norm_dist_prefix_sum.data_ptr<scalar_t>();
    auto norm_dist_data = norm_dist.data_ptr<scalar_t>();
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      size_t dist_id = item_id.get_id(0);
      size_t linear_id = item_id.get_linear_id();
      long* result_ptr = result_data;
      scalar_t* norm_dist_prefix_sum_ptr = norm_dist_prefix_sum_data;
      scalar_t* norm_dist_ptr = norm_dist_data;

      RandomState<Philox4_32_10> state(seeds.first, linear_id, seeds.second);
      float rand = state.uniform<float>();
      scalar_t r = ScalarConvert<float, scalar_t>::to(rand);

      // Find the bucket that a uniform sample lies in
      int choice = binary_search_for_multinomial<scalar_t>(
          norm_dist_prefix_sum_ptr + dist_id * categories,
          norm_dist_ptr + dist_id * categories,
          categories,
          r);

      // Torch indices are 1-based
      result_ptr[dist_id * num_samples + sample] = choice;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      norm_dist_ptr[dist_id * categories + choice] =
          ScalarConvert<int, scalar_t>::to(0);
    };

    cgh.parallel_for(range, kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// TODO: FLT_MANT_DIG is 24?
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (24);

Tensor& multinomial_out(
    const Tensor& self_original,
    int64_t num_samples,
    bool replacement,
    c10::optional<Generator> generator,
    Tensor& result) {
  TORCH_CHECK(
      result.device() == self_original.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self_original.dim() > 0 && self_original.dim() <= 2,
      "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self_original.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self_original.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ",
      result.scalar_type());
  TORCH_CHECK(num_samples > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self_original.size(-1);
  TORCH_CHECK(
      replacement || (num_samples <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  auto gen = get_generator_or_default<DPCPPGeneratorImpl>(
      generator, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  auto shape = self_original.sizes();
  auto num_dists = shape.size() == 1 ? 1 : shape[0];
  auto num_categories = shape.size() == 1 ? shape[0] : shape[1];

  Tensor self = self_original.contiguous();
  result.resize_({num_dists, num_samples});

  if (shape.size() == 1) {
    // add one dimension
    self = self.view({1, num_categories});
  }

  // Normalizes the L1 norm of every row to 1;
  Tensor norm_dist = self.renorm(1, 0, 1.0);
  std::pair<uint64_t, uint64_t> rng_engine_inputs;

  if (replacement) {
    // Sample without replacement
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);

      // each thread will utilize one random.
      rng_engine_inputs = gen->philox_engine_inputs(1);
    }

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    // Prefix sum along rows
    Tensor prefix_sum = norm_dist.cumsum(1);

    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        self.scalar_type(), "multinomial", [&] {
          sample_multinomial_with_replacement<scalar_t>(
              rng_engine_inputs,
              num_samples,
              num_dists,
              num_categories,
              result,
              prefix_sum,
              norm_dist);
        });
  } else {
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

    // Sample with replacement
    // we will use the same uniform distribution to draw the multinomial.
    // Prefix sum along rows
    Tensor prefix_sum = norm_dist.cumsum(1);

    for (int sample = 0; sample < num_samples; ++sample) {
      if (sample > 0) {
        // Update probabilities
        // Renorm along rows
        norm_dist.renorm_(1, 0, 1.0);
        // Prefix sum along rows
        at::cumsum_out(prefix_sum, norm_dist, 1);
      }
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);

        // each thread will utilize one random.
        rng_engine_inputs = gen->philox_engine_inputs(1);
      }

      IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
          self.scalar_type(), "multinomial", [&] {
            // The kernel can only draw one sample before we have to
            // recalculate our distribution
            sample_multinomial_without_replacement<scalar_t>(
                rng_engine_inputs,
                num_samples,
                sample,
                num_dists,
                num_categories,
                result,
                prefix_sum,
                norm_dist);
          });
    }
  }

  if (shape.size() == 1) {
    // remove one dimension
    result.resize_({num_samples});
  }

  return result;
}

Tensor multinomial(
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<Generator> generator) {
  auto shape = self.sizes();
  TORCH_CHECK(
      shape.size() > 0 && shape.size() <= 2,
      "prob_dist must be 1 or 2 dims. got ",
      shape.size());
  auto num_dists = shape.size() == 1 ? 1 : shape[1];
  auto num_categories = shape.size() == 1 ? shape[0] : shape[1];
  TORCH_CHECK(
      num_samples <= num_categories || replacement,
      "cannot sample n_sample > prob_dist size samples without replacement");

  Tensor ret = at::empty({num_dists, num_samples}, self.options().dtype(kLong));

  at::AtenIpexTypeXPU::multinomial_out(
      self, num_samples, replacement, generator, ret);
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at
