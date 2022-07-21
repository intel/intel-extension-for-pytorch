#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "comm/zmath.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void div_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "div",
      [&]() {
        dpcpp_fast_mode_kernel_with_scalars(
            iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
      });
}

} // namespace impl

void div_floor_kernel(TensorIterator& iter);
void div_trunc_kernel(TensorIterator& iter);

Tensor& div_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  impl::div_kernel_dpcpp(iter);
  return result;
}

Tensor& div_out(
    const Tensor& self,
    const Tensor& other,
    c10::optional<c10::string_view> rounding_mode,
    Tensor& result) {
  if (!rounding_mode.has_value()) {
    auto iter = TensorIterator::binary_float_op(result, self, other);
    impl::div_kernel_dpcpp(iter);
  } else if (*rounding_mode == "trunc") {
    auto iter = TensorIterator::binary_op(result, self, other);
    at::AtenIpexTypeXPU::div_trunc_kernel(iter);
  } else if (*rounding_mode == "floor") {
    auto iter = TensorIterator::binary_op(result, self, other);
    at::AtenIpexTypeXPU::div_floor_kernel(iter);
  }
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  impl::div_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::div_out(self, other, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
