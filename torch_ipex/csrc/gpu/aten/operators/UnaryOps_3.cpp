#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/AccumulateType.h>
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

Tensor clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::clamp_out(result, self, min, max);
}

Tensor& reciprocal_out(Tensor& out, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "reciprocal", [&] {
    using acc_t = acc_type<scalar_t>;
    if (at::dpcpp::TensorImpl_Unwrap(out) ==
        at::dpcpp::TensorImpl_Unwrap(self)) {
      at::dpcpp::DPCPP_tensor_apply1<scalar_t>(
          out, TensorReciprocalOp<scalar_t, acc_t>());
    } else {
      at::AtenIpexTypeDPCPP::resize_as_(out, self, c10::nullopt);
      at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
          out, self, TensorReciprocalOp<scalar_t, acc_t>());
    }
  });
  return out;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
