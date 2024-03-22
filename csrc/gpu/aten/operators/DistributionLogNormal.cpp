#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename accscalar_t>
struct log_normal_functor {
  scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(Numerics<accscalar_t>::exp(rand * std + mean));
  }
  log_normal_functor(accscalar_t mean, accscalar_t std)
      : mean(mean), std(std) {}

 private:
  accscalar_t mean;
  accscalar_t std;
};

Tensor& log_normal_(
    Tensor& self,
    double mean_,
    double std_,
    c10::optional<Generator> gen_) {
  TORCH_CHECK(
      std_ > 0.0, "log_normal_ expects std > 0.0, but found std=", std_);
  auto iter = TensorIterator::nullary_op(self);
  auto gen =
      get_generator_or_default<torch_ipex::xpu::dpcpp::DPCPPGeneratorImpl>(
          gen_, torch_ipex::xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "log_normal_dpcpp",
      [&] {
        using accscalar_t = dist_acctype<scalar_t>;
        auto mean = static_cast<accscalar_t>(mean_);
        auto std = static_cast<accscalar_t>(std_);
        // define functor to multiply std and add mean
        log_normal_functor<scalar_t, accscalar_t> log_normal_func(mean, std);
        normal_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, log_normal_func);
      });

  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
