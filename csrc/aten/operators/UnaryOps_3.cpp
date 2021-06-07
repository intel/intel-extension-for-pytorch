#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include <ATen/AtenIpexTypeXPU.h>

#include "Loops.h"


using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_ALL_CALLABLE_1_UNARY_OPS(clamp_max_, TensorMinValueOp);
IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(clamp_max_out, TensorMinValueOp);
IPEX_ALL_CALLABLE_1_UNARY_OPS(clamp_min_, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(clamp_min_out, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_2_UNARY_OPS(clamp_min_max, TensorClampOp);

Tensor& clamp_out(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (min && max) {
    at::AtenIpexTypeXPU::clamp_min_max(result, self, *min, *max);
  } else if (max) {
    at::AtenIpexTypeXPU::clamp_max_out(result, self, *max);
  } else if (min) {
    at::AtenIpexTypeXPU::clamp_min_out(result, self, *min);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeXPU::clamp_out(self, self, min, max);
}

Tensor clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::clamp_out(result, self, min, max);
}


class SyclOpReciprocal {};

Tensor& reciprocal_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    self.scalar_type(), 
    "reciprocal_xpu", [&] {
    dpcpp_kernel_for_tensor_iter<SyclOpReciprocal>(
        iter, [=](scalar_t a) -> scalar_t { return static_cast<scalar_t>(1.0) / a; });
  });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
