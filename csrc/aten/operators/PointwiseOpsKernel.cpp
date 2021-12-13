#include <ATen/Context.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void addcmul_kernel(TensorIterator& iter, Scalar value) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "addcmul_dpcpp",
      [&]() {
        auto alpha = value.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter, [alpha](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
              return a + alpha * b * c;
            });
      });
}

static void addcdiv_kernel(TensorIterator& iter, Scalar value) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "addcdiv_dpcpp",
      [&]() {
        auto alpha = value.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter, [alpha](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
              return a + alpha * (b / c);
            });
      });
}

} // namespace impl

Tensor& addcmul_out(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  // checkBackend("addcmul_cpu", out, self.options().backend());
  auto iter = at::TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .add_input(tensor1)
                  .add_input(tensor2)
                  .build();
  impl::addcmul_kernel(iter, value);
  return out;
}

Tensor addcmul(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::addcmul_out(
      self, tensor1, tensor2, value, result);
}

Tensor& addcmul_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  return at::AtenIpexTypeXPU::addcmul_out(self, tensor1, tensor2, value, self);
}

Tensor& addcdiv_out(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  // checkBackend("addcdiv_cpu", out, self.options().backend());
  if (isIntegralType(tensor1.scalar_type(), /*includeBool=*/true) &&
      isIntegralType(tensor2.scalar_type(), /*includeBool=*/true)) {
    TORCH_CHECK(
        false,
        "Integer division with addcdiv is no longer supported, and in a future  ",
        "release addcdiv will perform a true division of tensor1 and tensor2. ",
        "The historic addcdiv behavior can be implemented as ",
        "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
        "for integer inputs and as ",
        "(input + value * tensor1 / tensor2) for float inputs. ",
        "The future addcdiv behavior is just the latter implementation: ",
        "(input + value * tensor1 / tensor2), for all dtypes.");
  }
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .add_input(tensor1)
                  .add_input(tensor2)
                  .build();
  impl::addcdiv_kernel(iter, value);
  return out;
}

Tensor addcdiv(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::addcdiv_out(
      self, tensor1, tensor2, value, result);
}

Tensor& addcdiv_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  return at::AtenIpexTypeXPU::addcdiv_out(self, tensor1, tensor2, value, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
