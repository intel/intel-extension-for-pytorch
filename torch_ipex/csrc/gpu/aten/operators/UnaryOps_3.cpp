#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>
#include <utils/Pairwise.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

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
    at::AtenIpexTypeDPCPP::clamp_min_max(result, self, *min, *max);
  } else if (max) {
    at::AtenIpexTypeDPCPP::clamp_max_out(result, self, *max);
  } else if (min) {
    at::AtenIpexTypeDPCPP::clamp_min_out(result, self, *min);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeDPCPP::clamp_out(self, self, min, max);
}

IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(__and___out, TensorBitAndConstantOp);

Tensor __and__(const Tensor& self, Scalar other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__and___out(result, self, other);
}

Tensor& __iand__(Tensor& self, Scalar other) {
  return at::AtenIpexTypeDPCPP::__and___out(self, self, other);
}

IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(__or___out, TensorBitOrConstantOp);

Tensor __or__(const Tensor& self, Scalar other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__or___out(result, self, other);
}

Tensor& __ior__(Tensor& self, Scalar other) {
  return at::AtenIpexTypeDPCPP::__or___out(self, self, other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
