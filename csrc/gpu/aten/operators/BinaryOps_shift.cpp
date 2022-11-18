#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

#include "oneapi/dpl/type_traits"
namespace dpl = oneapi::dpl;

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& bitwise_left_shift_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift", [&]() {
    dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
      return static_cast<dpl::make_unsigned_t<scalar_t>>(a) << b;
    });
  });
  return out;
}

Tensor& bitwise_left_shift_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& result) {
  return at::bitwise_left_shift_out(
      result, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_left_shift(const Tensor& self, const Scalar& other) {
  return at::bitwise_left_shift(
      self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor& bitwise_left_shift_(Tensor& self, const Scalar& other) {
  return at::bitwise_left_shift_out(
      self, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_left_shift(const Scalar& self, const Tensor& other) {
  return at::bitwise_left_shift(
      wrapped_scalar_tensor(self).toType(other.scalar_type()), other);
}

Tensor& bitwise_right_shift_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift", [&]() {
    dpcpp_kernel_with_scalars(
        iter, [](scalar_t a, scalar_t b) -> scalar_t { return a >> b; });
  });
  return out;
}

Tensor& bitwise_right_shift_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& result) {
  return at::bitwise_right_shift_out(
      result, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_right_shift(const Tensor& self, const Scalar& other) {
  return at::bitwise_right_shift(
      self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor& bitwise_right_shift_(Tensor& self, const Scalar& other) {
  return at::bitwise_right_shift_out(
      self, self, wrapped_scalar_tensor(other).toType(self.scalar_type()));
}

Tensor bitwise_right_shift(const Scalar& self, const Tensor& other) {
  return at::bitwise_right_shift(
      wrapped_scalar_tensor(self).toType(other.scalar_type()), other);
}

Tensor __lshift__(const Tensor& self, const Tensor& other) {
  return at::bitwise_left_shift(self, other);
}

Tensor __lshift__(const Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  return at::bitwise_left_shift(self, wrapper);
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  return at::bitwise_left_shift_out(self, self, other);
}

Tensor& __ilshift__(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  return at::bitwise_left_shift_out(self, self, wrapper);
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  return at::bitwise_right_shift(self, other);
}

Tensor __rshift__(const Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  return at::bitwise_right_shift(self, wrapper);
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  return at::bitwise_right_shift_out(self, self, other);
}

Tensor& __irshift__(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  return at::bitwise_right_shift_out(self, self, wrapper);
}

} // namespace AtenIpexTypeXPU
} // namespace at
