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

void minimum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter(
        iter, [](bool a, bool b) -> bool { return a && b; });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_dpcpp", [&]() {
      dpcpp_kernel_for_tensor_iter(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "min_elementwise_dpcpp",
        [&]() {
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                if (a != a) {
                  return a;
                } else if (b != b) {
                  return b;
                } else {
                  return Numerics<scalar_t>::min(a, b);
                }
              });
        });
  }
}

void maximum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter(
        iter, [](bool a, bool b) -> bool { return a || b; });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "maximum_dpcpp", [&]() {
      dpcpp_kernel_for_tensor_iter(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t { return std::max(a, b); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "max_elementwise_dpcpp",
        [&]() {
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                if (a != a) {
                  return a;
                } else if (b != b) {
                  return b;
                } else {
                  return Numerics<scalar_t>::max(a, b);
                }
              });
        });
  }
}

} // namespace impl

Tensor& minimum_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "minimum does not support complex inputs.");

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::minimum_kernel(iter);
  return result;
}

Tensor minimum(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "minimum does not support complex inputs.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::minimum_kernel(iter);
  return iter.output();
}

// binary min, alias for minimum
Tensor& min_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::minimum_out(result, self, other);
}

Tensor& maximum_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "maximum does not support complex inputs.");

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::maximum_kernel(iter);
  return result;
}

Tensor maximum(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "maximum does not support complex inputs.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::maximum_kernel(iter);
  return iter.output();
}

// binary max, alias for maximum
Tensor& max_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::maximum_out(result, self, other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
