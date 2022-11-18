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

Tensor& log_normal_(
    Tensor& self,
    double mean_,
    double std_,
    c10::optional<Generator> gen_) {
  TORCH_CHECK(
      std_ > 0.0, "log_normal_ expects std > 0.0, but found std=", std_);
  auto iter = TensorIterator::nullary_op(self);
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "log_normal_dpcpp",
      [&] {
        using accscalar_t = dist_acctype<scalar_t>;
        auto mean = static_cast<accscalar_t>(mean_);
        auto std = static_cast<accscalar_t>(std_);
        // define lambda to multiply std and add mean
        auto log_normal_func = [mean, std](accscalar_t rand) {
          return static_cast<scalar_t>(
              Numerics<accscalar_t>::exp(rand * std + mean));
        };
        normal_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, log_normal_func);
      });

  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
