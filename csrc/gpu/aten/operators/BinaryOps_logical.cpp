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
namespace impl {

void logical_and_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      iter.common_dtype(),
      "logical_and_kernel",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter, [](scalar_t a, scalar_t b) -> bool { return a && b; });
      });
}

void logical_or_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.common_dtype(), "logical_or_kernel", [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter, [](scalar_t a, scalar_t b) -> bool { return a || b; });
      });
}

void logical_xor_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      iter.common_dtype(),
      "logical_xor_kernel",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter,
            [](scalar_t a, scalar_t b) -> bool { return bool(a) != bool(b); });
      });
}

void check_convert(Scalar scalar, ScalarType scalarType) {
  // Validate that is possible to convert scalar to tensor dtype without
  // overflow
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      scalarType,
      "check_convert",
      [&] { scalar.to<scalar_t>(); });
}

} // namespace impl

Tensor& logical_and_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      impl::check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      impl::check_convert(self.item(), other.scalar_type());
    }
  }
  auto iter = TensorIterator::comparison_op(result, self, other);
  impl::logical_and_kernel_dpcpp(iter);
  return result;
}

Tensor& logical_or_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      impl::check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      impl::check_convert(self.item(), other.scalar_type());
    }
  }
  auto iter = TensorIterator::comparison_op(result, self, other);
  impl::logical_or_kernel_dpcpp(iter);
  return result;
}

Tensor& logical_xor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      impl::check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      impl::check_convert(self.item(), other.scalar_type());
    }
  }
  auto iter = TensorIterator::comparison_op(result, self, other);
  impl::logical_xor_kernel_dpcpp(iter);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
