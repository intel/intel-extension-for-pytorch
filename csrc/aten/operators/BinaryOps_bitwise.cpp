#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void and_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_with_scalars(iter, [](bool a, bool b) { return a && b; });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_xpu", [&]() {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) { return a & b; });
    });
  }
}

static void or_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_with_scalars(iter, [](bool a, bool b) { return a || b; });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_or_xpu", [&]() {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) { return a | b; });
    });
  }
}

static void xor_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_with_scalars(iter, [](bool a, bool b) { return a != b; });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_xor_xpu", [&]() {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) { return a ^ b; });
    });
  }
}

} // namespace impl

Tensor& bitwise_and_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::and_kernel_dpcpp(iter);
  return out;
}

Tensor& bitwise_and_out(Tensor& out, const Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::bitwise_and_out(
      out, self, wrapped_scalar_tensor(other));
}

Tensor& bitwise_or_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::or_kernel_dpcpp(iter);
  return out;
}

Tensor& bitwise_or_out(Tensor& out, const Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::bitwise_or_out(
      out, self, wrapped_scalar_tensor(other));
}

Tensor& bitwise_xor_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::xor_kernel_dpcpp(iter);
  return out;
}

Tensor& bitwise_xor_out(Tensor& out, const Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::bitwise_xor_out(
      out, self, wrapped_scalar_tensor(other));
}

} // namespace AtenIpexTypeXPU
} // namespace at
