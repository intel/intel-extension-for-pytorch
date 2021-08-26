#include <ATen/ATen.h>
#include "Loops.h"
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

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
  auto scalarType =
      (iter.dtype() == ScalarType::Bool) ? iter.input_dtype() : iter.dtype();
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, scalarType, "logical_xor_kernel", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a, scalar_t b) -> scalar_t {
              return static_cast<scalar_t>(bool(a) != bool(b));
            });
      });

  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
