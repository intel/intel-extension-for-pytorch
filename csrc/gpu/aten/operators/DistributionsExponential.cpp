#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& exponential_(
    Tensor& self,
    double lambda,
    c10::optional<Generator> generator) {
  TORCH_CHECK(
      lambda >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambda);
  auto gen = get_generator_or_default<DPCPPGeneratorImpl>(
      generator, getDefaultDPCPPGenerator());
  auto iter = TensorIterator::nullary_op(self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "exponential_dpcpp_",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto lambd = static_cast<accscalar_t>(lambda);
        auto exponential_func = [lambd](accscalar_t rand) {
          auto sample =
              Numerics<accscalar_t>::log(static_cast<accscalar_t>(1.0) - rand);
          return static_cast<scalar_t>(
              static_cast<accscalar_t>(-1.0) / lambd * sample);
        };
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, exponential_func);
      });
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
