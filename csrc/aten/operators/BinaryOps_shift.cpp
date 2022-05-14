#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& bitwise_left_shift_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double ||
      iter.dtype() == ScalarType::Half ||
      iter.dtype() == ScalarType::BFloat16) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "lshift", [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return a * std::pow(static_cast<scalar_t>(2), b);
              });
        });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift", [&]() {
      dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
      });
    });
  }
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
  if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double ||
      iter.dtype() == ScalarType::Half ||
      iter.dtype() == ScalarType::BFloat16) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "rshift", [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return a / std::pow(static_cast<scalar_t>(2), b);
              });
        });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift", [&]() {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) -> scalar_t { return a >> b; });
    });
  }
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

} // namespace AtenIpexTypeXPU
} // namespace at
