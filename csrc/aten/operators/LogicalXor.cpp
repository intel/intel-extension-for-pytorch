#include <ATen/ATen.h>
#include "Loops.h"
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void logical_xor_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        kBool, kHalf, iter.input_dtype(), "logical_xor_kernel", [&]() {
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t a, scalar_t b) -> bool {
                return bool(a) != bool(b);
              });
        });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        kBool, kHalf, iter.dtype(), "logical_xor_kernel", [&]() {
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return static_cast<scalar_t>(bool(a) != bool(b));
              });
        });
  }
}

} // namespace impl

static void check_convert(Scalar scalar, ScalarType scalarType) {
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

Tensor& logical_xor_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      check_convert(self.item(), other.scalar_type());
    }
  }

  auto iter = TensorIterator::comparison_op(result, self, other);
  impl::logical_xor_kernel(iter);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
