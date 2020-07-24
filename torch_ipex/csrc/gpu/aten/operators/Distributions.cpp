#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <utils/AccumulateType.h>

#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#include "Random.h"
#include "Loops.h"

#ifdef USE_ONEMKL
#include <mkl_sycl.hpp>
#include <mkl.h>
#endif

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DPCPP_DEF_K1(bernoulli_tensor_dpcpp_ker);

DPCPP_DEF_K1(bernoulli_scalar_dpcpp_ker);

DPCPP_DEF_K1(bernoulli_tensor_sycl_random_filler);

DPCPP_DEF_K1(bernoulli_scalar_sycl_random_filler);

template <typename scalar_t>
void bernoulli_scalar_kernel(Tensor& ret, double p_, uint64_t seed) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto size = ret.numel() * sizeof(scalar_t);
  // First fill self with random number within range [0.0 - 1.0]
  auto cgf_1 = DPCPP_Q_CGF(cgh) {
    auto acc = DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size)
                   .get_access();
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = DPCPP::nd_range<1>(
        DPCPP::range<1>(global_range), DPCPP::range<1>(tile_size));
    cgh.parallel_for<DPCPP_K(bernoulli_scalar_sycl_random_filler, scalar_t)>(
        num_work_items, [=](cl::sycl::nd_item<1> item) {
          FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
          uniform_rnd_filler(item);
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf_1);

  // Generate final bernoulli distributions
  auto cgf_2 = DPCPP_Q_CGF(cgh) {
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto out_acc = DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>());
    cgh.parallel_for<DPCPP_K(bernoulli_scalar_dpcpp_ker, scalar_t)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_range), DPCPP::range<1>(tile_size)),
        [=](DPCPP::nd_item<1> item) {
          int64_t id = item.get_global_linear_id();
          auto out_ptr = out_acc.template get_pointer<scalar_t>();
          if (id < range) {
            out_ptr[id] = out_ptr[id] < p_ ? 1.0f : 0.0f;
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf_2);
}

template <typename scalar_t>
void bernoulli_tensor_kernel(Tensor& ret, const Tensor& p, uint64_t seed) {
  auto& queue = getCurrentDPCPPStream().dpcpp_queue();
  auto size = ret.numel() * sizeof(scalar_t);

  // First fill self with random number within range [0.0 - 1.0]
  auto cgf1 = DPCPP_Q_CGF(cgh) {
    auto acc = DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size)
                   .get_access();
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = DPCPP::nd_range<1>(
        DPCPP::range<1>(global_range), DPCPP::range<1>(tile_size));
    cgh.parallel_for<DPCPP_K(bernoulli_tensor_sycl_random_filler, scalar_t)>(
        num_work_items, [=](cl::sycl::nd_item<1> item) {
          FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
          uniform_rnd_filler(item);
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf1);

  // Generate final bernoulli distributions
  auto cgf2 = DPCPP_Q_CGF(cgh) {
    auto in_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, p.data_ptr<scalar_t>());
    auto out_acc =
        DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>());
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      int64_t id = item.get_global_linear_id();
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      if (id < range) {
        out_ptr[id] = out_ptr[id] < in_ptr[id] ? 1.0f : 0.0f;
      }
    };

    cgh.parallel_for<DPCPP_K(bernoulli_tensor_dpcpp_ker, scalar_t)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_range), DPCPP::range<1>(tile_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf2);
}

template <typename scalar_t>
int binary_search_for_multinomial(
    DPCPP::global_ptr<scalar_t> cumdist,
    DPCPP::global_ptr<scalar_t> dist,
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

DPCPP_DEF_K1(sample_multinomial_with_replacement_syck_ker);

template <typename scalar_t>
void sample_multinomial_with_replacement(
    std::pair<uint64_t, uint64_t> seeds,
    int num_samples,
    int64_t distributions,
    int categories,
    at::Tensor& result,
    at::Tensor& norm_dist_prefix_sum,
    at::Tensor& norm_dist) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  size_t work_group_size = dpcppMaxWorkGroupSize(dpcpp_queue);
  // Each work item in a work group will generate a sample from one
  // distribution concurrently
  work_group_size = std::min(work_group_size, (size_t)num_samples);
  DPCPP::range<2> global_range(distributions, work_group_size),
      local_range(1, work_group_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto result_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, result.data_ptr());
    auto norm_dist_prefix_sum_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, norm_dist_prefix_sum.data_ptr());
    auto norm_dist_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, norm_dist.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item_id) {
      size_t dist_id = item_id.get_group(0);
      size_t sample_id = item_id.get_local_id(1);
      size_t global_linear_id = item_id.get_global_linear_id();
      long* result_data = result_acc.template get_pointer<long>();
      scalar_t* norm_dist_prefix_sum_data =
          norm_dist_prefix_sum_acc.template get_pointer<scalar_t>();
      scalar_t* norm_dist_data = norm_dist_acc.template get_pointer<scalar_t>();

      RandomState<Philox4_32_10> state(
          seeds.first, global_linear_id, seeds.second);

      for (int sample = sample_id; sample < num_samples;
           sample += work_group_size) {
        float rand = state.uniform();
        scalar_t r = ScalarConvert<float, scalar_t>::to(rand);

        // Find the bucket that a uniform sample lies in
        int choice = binary_search_for_multinomial<scalar_t>(
            norm_dist_prefix_sum_data + dist_id * categories,
            norm_dist_data + dist_id * categories,
            categories,
            r);

        // Torch indices are 1-based
        result_data[dist_id * num_samples + sample] = choice;
      }
    };

    cgh.parallel_for<DPCPP_K(
        sample_multinomial_with_replacement_syck_ker, scalar_t)>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

DPCPP_DEF_K1(sample_multinomial_without_replacement_syck_ker);

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
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  size_t work_group_size = dpcppMaxWorkGroupSize(dpcpp_queue);
  // Each work item in a work group will generate a sample from one
  // distribution concurrently
  work_group_size = std::min(work_group_size, (size_t)num_samples);
  DPCPP::range<1> range(distributions);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto result_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, result.data_ptr());
    auto norm_dist_prefix_sum_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, norm_dist_prefix_sum.data_ptr());
    auto norm_dist_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, norm_dist.data_ptr());
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      size_t dist_id = item_id.get_id(0);
      size_t linear_id = item_id.get_linear_id();
      long* result_data = result_acc.template get_pointer<long>();
      scalar_t* norm_dist_prefix_sum_data =
          norm_dist_prefix_sum_acc.template get_pointer<scalar_t>();
      scalar_t* norm_dist_data = norm_dist_acc.template get_pointer<scalar_t>();

      RandomState<Philox4_32_10> state(seeds.first, linear_id, seeds.second);
      float rand = state.uniform();
      scalar_t r = ScalarConvert<float, scalar_t>::to(rand);

      // Find the bucket that a uniform sample lies in
      int choice = binary_search_for_multinomial<scalar_t>(
          norm_dist_prefix_sum_data + dist_id * categories,
          norm_dist_data + dist_id * categories,
          categories,
          r);

      // Torch indices are 1-based
      result_data[dist_id * num_samples + sample] = choice;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      norm_dist_data[dist_id * categories + choice] =
          ScalarConvert<int, scalar_t>::to(0);
    };

    cgh.parallel_for<DPCPP_K(
        sample_multinomial_without_replacement_syck_ker, scalar_t)>(range, kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
class exponential_sycl_ker {};
template<typename scalar_t, typename accscalar_t, typename dist_t, typename transform_t>
void distribution_elementwise_grid_stride_kernel(at::TensorIterator& iter,
                                                 int numel,
                                                 std::pair<uint64_t, uint64_t> seeds,
                                                 const dist_t dist_func,
                                                 const transform_t transform_func) {

  auto &sycl_queue = dpcpp::getCurrentDPCPPStream().dpcpp_queue();
  void* out_data = (void*)iter.data_ptr(0);

  using out_accessor_t = dpcpp::DPCPPAccessor<dpcpp_discard_w_mode>;
  auto offset_calc = at::dpcpp::make_offset_calculator<1>(iter);
  auto cgf = DPCPP_Q_CGF(cgh) {
    out_accessor_t out_acc = out_accessor_t (cgh, out_data);
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id)  {
      size_t sample_id = item_id.get_id(0);
      auto out_ptr = out_acc.template get_pointer<char>();
      RandomState<Philox4_32_10> state(seeds.first, sample_id, seeds.second);
      dist_func(&state);
      float rand = state.uniform();
      accscalar_t r = ScalarConvert<float, accscalar_t>::to(rand);
      scalar_t ret = transform_func(r);
      auto offset = offset_calc.get(sample_id)[0];
      scalar_t* out = (scalar_t*)(out_ptr + offset);
      *out = ret;
    };

    cgh.parallel_for<exponential_sycl_ker<scalar_t>>(
      DPCPP::range<1>(numel),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

template<typename scalar_t,
  typename accscalar_t,
  typename dist_t,
  typename transform_t>
void distribution_nullary_kernel(at::TensorIterator& iter,
                                 at::DPCPPGenerator* gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(1);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t>(sub_iter,
                                                         gen,
                                                         dist_func,
                                                         transform_func);
    }
    return;
  }

  distribution_elementwise_grid_stride_kernel<scalar_t, accscalar_t>(
    iter,
    numel,
    rng_engine_inputs,
    dist_func,
    transform_func);
}

} // namespace impl

Tensor& bernoulli_(Tensor& self, const Tensor& p_, Generator* _generator) {
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // Call dpcpp kernel to generate bernoulli distribution
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      p_.scalar_type(),
      "bernoulli_tensor_kernel",
      [&] {
        impl::bernoulli_tensor_kernel<scalar_t>(self, p_, gen->current_seed());
      });
  return self;
}

Tensor& bernoulli_(Tensor& self, double p, Generator* _generator) {
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // Call dpcpp kernel to generate bernoulli distribution
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "bernoulli_scalar_kernel",
      [&] {
        impl::bernoulli_scalar_kernel<scalar_t>(self, p, gen->current_seed());
      });
  return self;
}

Tensor& multinomial_out(
    Tensor& result,
    const Tensor& self_,
    int64_t num_samples,
    bool replacement,
    Generator* gen_) {
  auto gen = get_generator_or_default<DPCPPGenerator>(
      gen_, getDefaultDPCPPGenerator());
  auto shape = self_.sizes();
  TORCH_CHECK(
      shape.size() > 0 && shape.size() <= 2,
      "prob_dist must be 1 or 2 dims. got ",
      shape.size());
  auto num_dists = shape.size() == 1 ? 1 : shape[0];
  auto num_categories = shape.size() == 1 ? shape[0] : shape[1];
  TORCH_CHECK(
      num_samples <= num_categories || replacement,
      "cannot sample n_sample > prob_dist size samples without replacement");
  Tensor self = self_.contiguous();
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

    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "multinomial", [&] {
      impl::sample_multinomial_with_replacement<scalar_t>(
          rng_engine_inputs,
          num_samples,
          num_dists,
          num_categories,
          result,
          prefix_sum,
          norm_dist);
    });
  } else {
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

      IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "multinomial", [&] {
        // The kernel can only draw one sample before we have to
        // recalculate our distribution
        impl::sample_multinomial_without_replacement<scalar_t>(
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
    Generator* generator) {
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
  at::AtenIpexTypeDPCPP::multinomial_out(
      ret, self, num_samples, replacement, generator);
  return ret;
}

Tensor& exponential_(Tensor& self, double lambda_, Generator* gen_) {
  auto gen = get_generator_or_default<DPCPPGenerator>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());
#ifdef USE_ONEMKL
  if (lambda_ > 0) {
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "exponential_dpcpp_", [&] {
      scalar_t displ = static_cast<scalar_t>(0.0);
      scalar_t scale = static_cast<scalar_t>(std::abs(1/lambda_));
      auto &sycl_queue = dpcpp::getCurrentDPCPPStream().dpcpp_queue();
      uint64_t seed;
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        seed = gen->current_seed();
      }
      mkl::rng::philox4x32x10 engine(sycl_queue, seed);
      mkl::rng::exponential<scalar_t> distribution(displ, scale);
      auto sycl_buffer = make_buffer<scalar_t>(self.data_ptr());
      mkl::rng::generate(distribution, engine, self.numel(), sycl_buffer);
    });
  } else
#endif
  {
    auto iter = TensorIterator::nullary_op(self);
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "exponential_dpcpp_", [&] {
      using accscalar_t = acc_type<scalar_t>;
      auto lambda = static_cast<accscalar_t>(lambda_);

      // define lambda for exponential transformation
      auto exponential_func = [lambda] (accscalar_t rand) {
        accscalar_t sample;
        sample = cl::sycl::log(rand);
        return static_cast<scalar_t>(static_cast<accscalar_t>(-1.0) / lambda * sample);
      };
      impl::distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
                                                         gen,
                                                         [] (RandomState<Philox4_32_10>* state) {
                                                           return state->uniform(); },
                                                         exponential_func);

    });
  }

  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
