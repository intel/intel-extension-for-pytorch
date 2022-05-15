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

void maximum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_with_scalars(
        iter, [](bool a, bool b) -> bool { return a || b; });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "max_elementwise_dpcpp",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return Numerics<scalar_t>::max(a, b);
              });
        });
  }
}

} // namespace impl

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

Tensor& fmax_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  if (isFloatingType(iter.common_dtype())) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmax",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return Numerics<scalar_t>::fmax(a, b);
              });
        });
  } else {
    impl::maximum_kernel(iter);
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
