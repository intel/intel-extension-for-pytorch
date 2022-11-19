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

static void mul_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "mul",
      [&]() {
        dpcpp_fast_mode_kernel_with_scalars(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; });
      });
}
} // namespace impl

Tensor& mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::mul_kernel_dpcpp(iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::mul_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::mul_out(self, other, self);
}

Tensor mul(const Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::mul(self, wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::mul_out(self, wrapped_scalar_tensor(other), self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
