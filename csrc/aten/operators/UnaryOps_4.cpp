#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include <ATen/AtenIpexTypeXPU.h>

#include "comm/zmath.h"
#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(tanh_out, Numerics<scalar_t>::tanh, Real);

Tensor& tanh_(Tensor& self) {
  return at::AtenIpexTypeXPU::tanh_out(self, self);
}

Tensor tanh(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::tanh_out(result, self);
}

IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(erfinv_out, TensorErfinvOp);

Tensor& erfinv_(Tensor& self) {
  return at::AtenIpexTypeXPU::erfinv_out(self, self);
}

Tensor erfinv(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::erfinv_out(result, self);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(remainder_out, TensorRemainderOp);

Tensor remainder(const Tensor& self, Scalar other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(fmod_out, TensorFmodOp);

Tensor fmod(const Tensor& self, Scalar other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_UNARY_OPS(sign_out, TensorSignOp);

Tensor sign(const Tensor& self) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::sign_out(out, self);
}

Tensor& sign_(Tensor& self) {
  return at::AtenIpexTypeXPU::sign_out(self, self);
}

class SyclOpConj {};
Tensor& conj_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  // IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "conj_xpu", [&]() {
  IPEX_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "conj_xpu", [&]() {
      dpcpp_kernel_for_tensor_iter<SyclOpConj>(
          iter, [=](scalar_t a) -> scalar_t { return conj_impl(a); });
  });

  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
