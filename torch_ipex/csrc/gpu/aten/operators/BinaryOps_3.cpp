#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/General.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
typename std::enable_if<IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t), void>::type
__and___out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (at::dpcpp::TensorImpl_Unwrap(result) ==
      at::dpcpp::TensorImpl_Unwrap(self)) {
    at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        result, other, TensorBitAndOp<scalar_t>());
  } else {
    at::AtenIpexTypeDPCPP::resize_as_(result, self, c10::nullopt);
    at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        result, self, other, TensorBitAndOp<scalar_t>());
  }
}

template <typename scalar_t>
typename std::enable_if<!(IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t)), void>::
    type
    __and___out(Tensor& result, const Tensor& self, const Tensor& other) {}

template <typename scalar_t>
typename std::enable_if<IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t), void>::type
__or___out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (at::dpcpp::TensorImpl_Unwrap(result) ==
      at::dpcpp::TensorImpl_Unwrap(self)) {
    at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        result, other, TensorBitOrOp<scalar_t>());
  } else {
    at::AtenIpexTypeDPCPP::resize_as_(result, self, c10::nullopt);
    at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        result, self, other, TensorBitOrOp<scalar_t>());
  }
}

template <typename scalar_t>
typename std::enable_if<!(IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t)), void>::
    type
    __or___out(Tensor& result, const Tensor& self, const Tensor& other) {}

} // namespace impl

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(min_out, TensorMinOp)

Tensor min(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::min_out(out, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(max_out, TensorMaxOp)

Tensor max(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::max_out(out, self, other);
}

Tensor& bitwise_and_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, self.scalar_type(), "__and___out", [&]() {
        impl::__and___out<scalar_t>(result, self, other);
      });
  return result;
}

Tensor& bitwise_or_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, self.scalar_type(), "__or___out", [&]() {
        impl::__or___out<scalar_t>(result, self, other);
      });
  return result;
}

Tensor& bitwise_and_out(Tensor& out, const Tensor& self, Scalar other) {
  auto other_ = c10::scalar_to_tensor(other, kDPCPP);
  // TODO: broadcast
  auto new_other =
      other_.resize_as_(self).fill_(other).toType(self.scalar_type());
  return at::AtenIpexTypeDPCPP::bitwise_and_out(out, self, new_other);
}

Tensor& bitwise_or_out(Tensor& out, const Tensor& self, Scalar other) {
  auto other_ = c10::scalar_to_tensor(other, kDPCPP);
  // TODO: broadcast
  auto new_other =
      other_.resize_as_(self).fill_(other).toType(self.scalar_type());
  return at::AtenIpexTypeDPCPP::bitwise_or_out(out, self, new_other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
