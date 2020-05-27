#include <ATen/ATen.h>
#include <utils/AccumulateType.h>

#include <core/Context.h>
#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <utils/Numerics.h>
#include <utils/Pointwise.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
void sigmoid(Tensor& output, const Tensor& self) {
  if (TensorImpl_Unwrap(output) == TensorImpl_Unwrap(self)) {
    at::dpcpp::DPCPP_tensor_apply1<scalar_t>(
        output, TensorSigmoidOp<scalar_t>());
  } else {
    output.resize_as_(self);
    at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        output, self, TensorSigmoidOp<scalar_t>());
  }
}

Tensor& _sigmoid_out(Tensor& output, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_sigmoid_out",
      [&]() { impl::sigmoid<scalar_t>(output, self); });
  return output;
}

} // namespace impl

Tensor& sigmoid_out(Tensor& out, const Tensor& self) {
  return impl::_sigmoid_out(out, self);
}
Tensor sigmoid(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::sigmoid_out(result, self);
}
Tensor& sigmoid_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::sigmoid_out(self, self);
}

Tensor& sigmoid_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output) {
  TORCH_CHECK(output.numel() == grad_output.numel(), "different elements ...");
  grad_input.resize_as_(output);
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      output.scalar_type(),
      "sigmoid_backward_out",
      [&]() {
        at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
            grad_input, output, grad_output, TensorSigmoidGradOp<scalar_t>());
      });

  return grad_input;
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  auto grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeDPCPP::sigmoid_backward_out(
      grad_input, grad_output, output);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
