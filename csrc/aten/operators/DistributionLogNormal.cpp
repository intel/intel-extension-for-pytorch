#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include <runtime/DPCPPUtils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"

#include "Random.h"
#include "Distributions.h"

#ifdef USE_ONEMKL
#include <oneapi/mkl.hpp>
#include <utils/oneMKLUtils.h>
#endif

namespace at {
namespace AtenIpexTypeXPU {

void log_normal_dpcpp(TensorIterator& iter, double mean_, double std_, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_dpcpp", [&] {
    using accscalar_t = dist_acctype<scalar_t>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto log_normal_func = [mean, std](accscalar_t rand) {
      return static_cast<scalar_t>(DPCPP::exp(rand * std + mean));
    };
    AtenIpexTypeXPU::distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
      gen,
      [](RandomState<Philox4_32_10> *state) { return state->normal<scalar_t>(); },
      log_normal_func);
  });
}


Tensor& log_normal_(Tensor& self, double mean_, double std_, c10::optional<Generator> gen_) {
  TORCH_CHECK(std_ > 0.0, "log_normal_ expects std > 0.0, but found std=", std_);
#ifdef USE_ONEMKL
  if (self.is_contiguous()) {
    auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());

    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "log_normal_", [&] {
      auto mean = static_cast<scalar_t>(mean_);
      auto std = static_cast<scalar_t>(std_);
      scalar_t displ = static_cast<scalar_t>(0.0);
      scalar_t scale = static_cast<scalar_t>(1.0);
      auto &dpcpp_queue = xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue();
      oneapi::mkl::rng::philox4x32x10 engine(dpcpp_queue, gen->seed());
      oneapi::mkl::rng::lognormal<scalar_t, oneapi::mkl::rng::lognormal_method::box_muller2> distribution(mean, std, displ, scale);
      DPCPP_ONEMKL_SUBMIT(dpcpp_queue, oneapi::mkl::rng::generate, distribution, engine, self.numel(), (scalar_t *)(self.data_ptr()));
    });
  } else
#endif
  {
    auto iter = TensorIterator::nullary_op(self);
    log_normal_dpcpp(iter, mean_, std_, gen_);
  }
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
