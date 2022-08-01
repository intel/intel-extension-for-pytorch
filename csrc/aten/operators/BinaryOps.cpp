#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/ComplexHelper.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& heaviside_out(const Tensor& self, const Tensor& values, Tensor& out) {
  TORCH_CHECK(
      !self.is_complex() && !values.is_complex(),
      "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(
      self.dtype() == values.dtype(),
      "heaviside is not yet implemented for tensors with different dtypes.");

  auto iter = TensorIterator::binary_op(out, self, values);
  IPEX_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, kBFloat16, iter.dtype(), "heaviside", [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
          return a == 0 ? b : static_cast<scalar_t>(a > 0);
        });
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
