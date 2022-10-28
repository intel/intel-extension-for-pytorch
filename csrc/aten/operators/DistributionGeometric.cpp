#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

void geometric_scalar_dpcpp(
    TensorIterator& iter,
    double p_,
    c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "geometric_scalar_dpcpp",
      [&] {
        using accscalar_t = DiscreteDistributionType<scalar_t>::type;
        auto p = static_cast<accscalar_t>(p_);
        auto geometric_func = [p](accscalar_t rand) {
          // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
          return static_cast<scalar_t>(Numerics<accscalar_t>::ceil(
              at::log(rand) / at::log(static_cast<accscalar_t>(1.0) - p)));
        };
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, geometric_func);
      });
}

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen_) {
  auto iter = TensorIterator::nullary_op(self);
  geometric_scalar_dpcpp(iter, p, gen_);
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
