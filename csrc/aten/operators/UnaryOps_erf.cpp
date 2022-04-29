#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& erf_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "erf", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::erf(a);
        });
      });
  return out;
}

Tensor& erfc_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "erfc", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::erfc(a);
        });
      });
  return out;
}

Tensor& erfinv_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "erfinv", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          scalar_t b;
          TensorErfinvOp<scalar_t>()(b, a);
          return b;
        });
      });
  return out;
}

Tensor& erfinv_(Tensor& self) {
  return at::AtenIpexTypeXPU::erfinv_out(self, self);
}

Tensor erfinv(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::erfinv_out(result, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
