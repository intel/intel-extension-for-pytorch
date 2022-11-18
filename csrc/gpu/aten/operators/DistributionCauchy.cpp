#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor& cauchy_(
    Tensor& self,
    double median_,
    double sigma_,
    c10::optional<Generator> generator) {
  auto iter = TensorIterator::nullary_op(self);
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      generator, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "cauchy_dpcpp",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        auto median = static_cast<accscalar_t>(median_);
        auto sigma = static_cast<accscalar_t>(sigma_);
        auto cauchy_func = [median, sigma](accscalar_t rand) {
          return static_cast<scalar_t>(
              median +
              sigma *
                  at::tan(
                      Numerics<accscalar_t>::pi() *
                      (rand - static_cast<accscalar_t>(0.5))));
        };
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, cauchy_func);
      });
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
