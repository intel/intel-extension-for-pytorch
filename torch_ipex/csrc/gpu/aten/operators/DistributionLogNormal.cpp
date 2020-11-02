#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include <utils/ATDispatch.h>
#include <utils/AccumulateType.h>

#include "Random.h"
#include "Distributions.h"

#ifdef USE_ONEMKL
#include <mkl_sycl.hpp>
#endif

namespace at {
namespace AtenIpexTypeDPCPP {


void log_normal_dpcpp(TensorIterator& iter, double mean_, double std_, Generator* gen_) {
  auto gen = get_generator_or_default<DPCPPGenerator>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_dpcpp", [&] {
    using accscalar_t = dist_acctype<scalar_t>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto log_normal_func = [mean, std](accscalar_t rand) {
      return static_cast<scalar_t>(DPCPP::exp(rand * std + mean));
    };
    AtenIpexTypeDPCPP::distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
      gen,
      [](RandomState<Philox4_32_10> *state) { return state->normal<scalar_t>(); },
      log_normal_func);
  });
}


Tensor& log_normal_(Tensor& self, double mean_, double std_, Generator* gen_) {
  TORCH_CHECK(std_ > 0.0, "log_normal_ expects std > 0.0, but found std=", std_);
#ifdef USE_ONEMKL
  if (self.is_contiguous()) {
    auto gen = get_generator_or_default<DPCPPGenerator>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());

    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "log_normal_", [&] {
      auto mean = static_cast<scalar_t>(mean_);
      auto std = static_cast<scalar_t>(std_);
      scalar_t displ = static_cast<scalar_t>(0.0);
      scalar_t scale = static_cast<scalar_t>(1.0);
      auto &dpcpp_queue = dpcpp::getCurrentDPCPPStream().dpcpp_queue();
      mkl::rng::philox4x32x10 engine(dpcpp_queue, gen->seed());
      mkl::rng::lognormal<scalar_t, mkl::rng::lognormal_method::box_muller2> distribution(mean, std, displ, scale);
      auto dpcpp_buffer = make_buffer<scalar_t>(self.data_ptr());
      mkl::rng::generate(distribution, engine, self.numel(), dpcpp_buffer);
    });
  } else
#endif
  {
    auto iter = TensorIterator::nullary_op(self);
    log_normal_dpcpp(iter, mean_, std_, gen_);
  }
  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
