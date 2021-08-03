#include <ATen/ATen.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename...>
class TensorSigmoidOp {};
template <typename...>
class TensorSigmoidGradOp {};

template <typename scalar_t>
void sigmoid(Tensor& output, const Tensor& self) {
  output.resize_as_(self);
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(output)
                  .add_input(self)
                  .build();
  dpcpp_kernel_for_tensor_iter<TensorSigmoidOp<scalar_t>>(
      iter, [=](scalar_t v) -> scalar_t {
        scalar_t one = (scalar_t)1.0;
        return one / (one + Numerics<scalar_t>::exp(-static_cast<scalar_t>(v)));
      });
}

template <typename scalar_t>
void sigmoid_backward(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& self) {
  gradInput.resize_as_(self);
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(gradInput)
                  .add_input(gradOutput)
                  .add_input(self)
                  .build();
  if (iter.dtype() == ScalarType::Half) {
    dpcpp_kernel_for_tensor_iter<TensorSigmoidGradOp<at::Half>>(
        iter, [=](at::Half go, at::Half in) -> at::Half {
          float in_float = (float)in;
          float go_float = (float)go;
          return (at::Half)(go * (1.f - in_float) * in_float);
        });
  } else {
    dpcpp_kernel_for_tensor_iter<TensorSigmoidGradOp<scalar_t>>(
        iter, [=](scalar_t go, scalar_t in) -> scalar_t {
          scalar_t one = (scalar_t)1.0;
          return go * (one - in) * in;
        });
  }
}

Tensor& _sigmoid_out(Tensor& output, const Tensor& self) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_sigmoid_out",
      [&]() { impl::sigmoid<scalar_t>(output, self); });
  return output;
}

Tensor& _sigmoid_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_sigmoid_backward_out",
      [&]() {
        impl::sigmoid_backward<scalar_t>(grad_input, grad_output, self);
      });
  return grad_input;
}

} // namespace impl

Tensor& sigmoid_out(Tensor& out, const Tensor& self) {
  return impl::_sigmoid_out(out, self);
}
Tensor sigmoid(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::sigmoid_out(result, self);
}
Tensor& sigmoid_(Tensor& self) {
  return at::AtenIpexTypeXPU::sigmoid_out(self, self);
}

Tensor& sigmoid_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output) {
  TORCH_CHECK(output.numel() == grad_output.numel(), "different elements ...");
  return impl::_sigmoid_backward_out(grad_input, grad_output, output);
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  auto grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeXPU::sigmoid_backward_out(
      grad_input, grad_output, output);
}

} // namespace AtenIpexTypeXPU
} // namespace at
