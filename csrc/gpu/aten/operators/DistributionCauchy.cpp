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

template <typename scalar_t, typename accscalar_t>
struct cauchy_functor {
  scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(
        median +
        sigma *
            Numerics<accscalar_t>::tan(
                Numerics<accscalar_t>::pi() *
                (rand - static_cast<accscalar_t>(0.5))));
  }

  cauchy_functor(accscalar_t median, accscalar_t sigma)
      : median(median), sigma(sigma) {}

 private:
  accscalar_t median;
  accscalar_t sigma;
};

Tensor& cauchy_(
    Tensor& self,
    double median_,
    double sigma_,
    c10::optional<Generator> generator) {
  TORCH_CHECK(
      sigma_ > 0.0, "cauchy_ expects sigma > 0.0, but found sigma=", sigma_);
  auto iter = TensorIterator::nullary_op(self);
  auto gen =
      get_generator_or_default<torch_ipex::xpu::dpcpp::DPCPPGeneratorImpl>(
          generator,
          torch_ipex::xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "cauchy_dpcpp",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        auto median = static_cast<accscalar_t>(median_);
        auto sigma = static_cast<accscalar_t>(sigma_);
        cauchy_functor<scalar_t, accscalar_t> cauchy_func(median, sigma);
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, cauchy_func);
      });
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
