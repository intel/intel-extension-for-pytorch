#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/AtenIpexTypeXPU.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_ALL_CALLABLE_1_CONST_UNARY_OPS(clamp_max_out, TensorMinValueOp);
IPEX_OUT_ALL_CALLABLE_1_CONST_UNARY_OPS(clamp_min_out, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_2_CONST_UNARY_OPS(clamp_min_max, TensorClampOp);

Tensor& clamp_out(
    const Tensor& self,
    const optional<Scalar>& min,
    const optional<Scalar>& max,
    Tensor& result) {
  if (min && max) {
    at::AtenIpexTypeXPU::clamp_min_max(self, *min, *max, result);
  } else if (max) {
    at::AtenIpexTypeXPU::clamp_max_out(self, *max, result);
  } else if (min) {
    at::AtenIpexTypeXPU::clamp_min_out(self, *min, result);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeXPU::clamp_out(self, min, max, self);
}

Tensor clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::clamp_out(self, min, max, result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
