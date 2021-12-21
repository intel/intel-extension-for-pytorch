#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/AtenIpexTypeXPU.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/zmath.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_ALL_UNARY_FUNC_OPS(abs_out, Numerics<scalar_t>::abs, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(neg_out, Numerics<scalar_t>::neg, Real);

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(round_out, Numerics<scalar_t>::round, Real);

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(remainder_out, TensorRemainderOp);

Tensor remainder(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(fmod_out, TensorFmodOp);

Tensor fmod(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

Tensor& conj_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  // IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(),
  // "conj_xpu", [&]() {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "conj_xpu", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t a) -> scalar_t { return conj_impl(a); });
      });

  return out;
}

Tensor& reciprocal_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "reciprocal_xpu",
      [&] {
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t a) -> scalar_t {
          return static_cast<scalar_t>(1.0) / a;
        });
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
