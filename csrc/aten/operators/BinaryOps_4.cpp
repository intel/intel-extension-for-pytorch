#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AtenIpexTypeXPU.h>

#include <runtime/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"

#include "Loops.h"


using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

DPCPP_DEF_K1(tanh_backward);
Tensor& tanh_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output) {
  auto iter = TensorIteratorConfig()
  .set_check_mem_overlap(true)
  .add_output(grad_input)
  .add_input(grad_output)
  .add_input(output)
  .build();

  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "tanh_backward_out", [&]() {
        dpcpp_kernel_for_tensor_iter<DPCPP_K(tanh_backward)>(
            iter, [](scalar_t output, scalar_t z) -> scalar_t {
              return output * (1. - z * z);
            });
      });

  return grad_input;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  auto grad_input = at::empty({0}, grad_output.options());
  return at::tanh_backward_out(grad_input, grad_output, output);
}

DPCPP_DEF_K1(atan2);
Tensor& atan2_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "atan2",
      [&]() {
        dpcpp_kernel_for_tensor_iter<DPCPP_K(atan2)>(
            iter, [](scalar_t a, scalar_t b) -> scalar_t {
              return Numerics<scalar_t>::atan2(a, b);
            });
      });
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::atan2_out(result, self, other);
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::atan2_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(remainder_out, TensorCRemainderOp)

Tensor remainder(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(fmod_out, TensorCFmodOp)

Tensor fmod(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
