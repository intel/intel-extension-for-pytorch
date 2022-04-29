#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void frac_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "frac_xpu", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return a - Numerics<scalar_t>::trunc(a);
        });
      });
}

} // namespace impl

Tensor& frac_out(Tensor& result, const Tensor& self) {
  auto iter = TensorIterator::unary_op(result, self);
  impl::frac_kernel(iter);
  return result;
}

Tensor& trunc_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "trunc_out",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::trunc(a);
        });
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
