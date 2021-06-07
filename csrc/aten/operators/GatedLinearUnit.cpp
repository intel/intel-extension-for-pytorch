#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>

#include "comm/Numerics.h"
#include "comm/ATDispatch.h"


using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void GatedLinearUnit_updateOutput(
    Tensor& output,
    const Tensor& input,
    int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nln = input.size(wrap_dim);
  TORCH_CHECK(
      nln % 2 == 0,
      "Halving dimension must be even, but dimension",
      wrap_dim,
      " is size ",
      nln);
  const int64_t inputSize = nln / 2;
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  // output = output + firstHalf * sigmoid(secondHalf)
  Tensor sigNum = at::empty_like(secondHalf);
  at::sigmoid_out(sigNum, secondHalf);
  output = at::mul(firstHalf, sigNum);
}

template <typename scalar_t>
void GatedLinearUnit_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    int64_t dim) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nln = input.size(wrap_dim);
  TORCH_CHECK(
      nln % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nln);

  grad_input.resize_as_(input);
  const int64_t inputSize = nln / 2;
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
  Tensor gradInputsecondHalf =
      grad_input.narrow(wrap_dim, inputSize, inputSize);

  // gradInputfirstHalf = grad_output * sigmoid(secondHalf)
  // gradInputsecondHalf = (1 - sigmoid(secondHalf)) * sigmoid(secondHalf) *
  // input * grad_output
  at::sigmoid_out(gradInputfirstHalf, secondHalf);
  gradInputsecondHalf.fill_(ScalarConvert<int, scalar_t>::to(1));
  gradInputsecondHalf.sub_(gradInputfirstHalf)
      .mul_(gradInputfirstHalf)
      .mul_(firstHalf);
  gradInputfirstHalf.mul_(grad_output);
  gradInputsecondHalf.mul_(grad_output);
}

} // namespace impl

// namespace AtenIpexTypeXPU
Tensor& glu_out(Tensor& out, const Tensor& self, int64_t dim) {
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "glu_out", [&] {
    impl::GatedLinearUnit_updateOutput<scalar_t>(out, self, dim);
  });
  return out;
}

Tensor glu(const Tensor& self, int64_t dim) {
  Tensor out = at::empty({}, self.options());
  return at::AtenIpexTypeXPU::glu_out(out, self, dim);
}

Tensor& glu_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim) {
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "glu_backward_out", [&] {
        impl::GatedLinearUnit_updateGradInput<scalar_t>(
            grad_input, grad_output, self, dim);
      });
  return grad_input;
}

Tensor glu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim) {
  Tensor grad_input = at::empty({}, self.options());
  return at::AtenIpexTypeXPU::glu_backward_out(
      grad_input, grad_output, self, dim);
}

} // namespace AtenIpexTypeXPU
} // namespace at
