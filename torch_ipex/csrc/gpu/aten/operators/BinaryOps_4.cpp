#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

DPCPP_DEF_K1(tanh_backward);
Tensor& tanh_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output) {
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(grad_output);
  iter.add_input(output);
  iter.build();

  AT_DISPATCH_ALL_TYPES(iter.dtype(), "tanh_backward_out", [&]() {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "atan2", [&]() {
    dpcpp_kernel_for_tensor_iter<DPCPP_K(atan2)>(
        iter, [](scalar_t a, scalar_t b) -> scalar_t {
          return Numerics<scalar_t>::atan2(a, b);
        });
  });
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::atan2_out(result, self, other);
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::atan2_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(remainder_out, TensorCRemainderOp)

Tensor remainder(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(fmod_out, TensorCFmodOp)

Tensor fmod(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::fmod_out(self, self, other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
