#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <ATen/xpu/XPUGeneratorImpl.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename accscalar_t>
struct exponential_functor {
  auto operator()(accscalar_t val) const {
    // BEFORE TOUCHING THIS CODE READ:
    // https://github.com/pytorch/pytorch/issues/16706
    // rand_uniform has (0,1] bounds. log(1) is 0 and exponential
    // excludes 0. we need log to be not 0, and not underflow when
    // converted to half
    accscalar_t log;
    if (val >= static_cast<accscalar_t>(1.f) -
            std::numeric_limits<scalar_t>::epsilon() / 2.f) {
      // Need an epsilon of appropriate precision.
      // Unlike CUDA behavior, DPCPP dtype conversions do not support
      // epsilon downgrading.
      log = -std::numeric_limits<scalar_t>::epsilon() / 2.f;
    } else {
      log = Numerics<accscalar_t>::log(val);
    }
    return static_cast<accscalar_t>(-1.f) / lambd * log;
  }

  exponential_functor(accscalar_t lambd) : lambd(lambd) {}

 private:
  accscalar_t lambd;
};

Tensor& exponential_(
    Tensor& self,
    double lambda,
    c10::optional<Generator> generator) {
  TORCH_CHECK(
      lambda >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambda);
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());
  auto iter = TensorIterator::nullary_op(self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "exponential_dpcpp_",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto lambd = static_cast<accscalar_t>(lambda);
        exponential_functor<scalar_t, accscalar_t> exponential_func(lambd);
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, exponential_func);
      });
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
