#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>
#include <utils/ATDispatch.h>
#include "Pooling.h"

using namespace dnnl;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void adaptive_max_pool3d_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool3d_dpcpp(): expected input to have non-empty spatial "
        "dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 3,
      "adaptive_max_pool3d: internal error: output_size.size() must be 3");

  Tensor input_ = input.contiguous();

  int64_t nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
  int64_t nblock = input_.size(-4);
  int64_t inputDepth = input_.size(-3);
  int64_t inputHeight = input_.size(-2);
  int64_t inputWidth = input_.size(-1);

  int64_t outputDepth = output_size[0];
  int64_t outputHeight = output_size[1];
  int64_t outputWidth = output_size[2];

  int dD = DPCPP::floor((float)2 * inputDepth / outputDepth) -
      DPCPP::floor((float)inputDepth / outputDepth);
  int dH = DPCPP::floor((float)2 * inputHeight / outputHeight) -
      DPCPP::floor((float)inputHeight / outputHeight);
  int dW = DPCPP::floor((float)2 * inputWidth / outputWidth) -
      DPCPP::floor((float)inputWidth / outputWidth);

  int kD = DPCPP::ceil((float)2 * inputDepth / outputDepth) -
      DPCPP::floor((float)inputDepth / outputDepth);
  int kH = DPCPP::ceil((float)2 * inputHeight / outputHeight) -
      DPCPP::floor((float)inputHeight / outputHeight);
  int kW = DPCPP::ceil((float)2 * inputWidth / outputWidth) -
      DPCPP::floor((float)inputWidth / outputWidth);

  int padD = (dD * (outputDepth - 1) + kD - inputDepth) / 2;
  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;

  auto alg_kind = algorithm::pooling_max;
  auto prop_kind = dnnl::prop_kind::forward_training;
  if (input_.ndimension() == 4) {
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
    indices.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
    indices.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
  }

  max_pool_out_frame<algorithm::pooling_max>(
      input_,
      output,
      indices,
      nbatch,
      nblock,
      inputDepth,
      inputHeight,
      inputWidth,
      outputDepth,
      outputHeight,
      outputWidth,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW);
}

Tensor& adaptive_max_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    const Tensor& indices) {
  Tensor gradOutput = gradOutput_.contiguous();

  int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  int64_t nblock = input.size(-4);
  int64_t gradInputDepth = input.size(-3);
  int64_t gradInputHeight = input.size(-2);
  int64_t gradInputWidth = input.size(-1);

  int64_t gradOutputDepth = gradOutput.size(-3);
  int64_t gradOutputHeight = gradOutput.size(-2);
  int64_t gradOutputWidth = gradOutput.size(-1);

  int dD = DPCPP::floor((float)2 * gradInputDepth / gradOutputDepth) -
      DPCPP::floor((float)gradInputDepth / gradOutputDepth);
  int dH = DPCPP::floor((float)2 * gradInputHeight / gradOutputHeight) -
      DPCPP::floor((float)gradInputHeight / gradOutputHeight);
  int dW = DPCPP::floor((float)2 * gradInputWidth / gradOutputWidth) -
      DPCPP::floor((float)gradInputWidth / gradOutputWidth);

  int kD = DPCPP::ceil((float)2 * gradInputDepth / gradOutputDepth) -
      DPCPP::floor((float)gradInputDepth / gradOutputDepth);
  int kH = DPCPP::ceil((float)2 * gradInputHeight / gradOutputHeight) -
      DPCPP::floor((float)gradInputHeight / gradOutputHeight);
  int kW = DPCPP::ceil((float)2 * gradInputWidth / gradOutputWidth) -
      DPCPP::floor((float)gradInputWidth / gradOutputWidth);

  int padD = (dD * (gradOutputDepth - 1) + kD - gradInputDepth) / 2;
  int padH = (dH * (gradOutputHeight - 1) + kH - gradInputHeight) / 2;
  int padW = (dW * (gradOutputWidth - 1) + kW - gradInputWidth) / 2;

  gradInput.resize_as_(input);
  gradInput.zero_();

  max_pool_backward_out_frame<algorithm::pooling_max>(
      gradInput,
      gradOutput,
      indices,
      nbatch,
      nblock,
      gradInputDepth,
      gradInputHeight,
      gradInputWidth,
      gradOutputDepth,
      gradOutputHeight,
      gradOutputWidth,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW);
  return gradInput;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> adaptive_max_pool3d_out(
    Tensor& out,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef output_size) {
  impl::adaptive_max_pool3d_out_template(out, indices, self, output_size);
  return std::tuple<Tensor&, Tensor&>(out, indices);
}

std::tuple<Tensor, Tensor> adaptive_max_pool3d(
    const Tensor& self,
    IntArrayRef output_size) {
  Tensor output = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::AtenIpexTypeXPU::adaptive_max_pool3d_out(
      output, indices, self, output_size);
}

Tensor& adaptive_max_pool3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  impl::adaptive_max_pool3d_backward_out_template(
      grad_input, grad_output, self, indices);
  return grad_input;
}

Tensor adaptive_max_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  auto grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  impl::adaptive_max_pool3d_backward_out_template(
      grad_input, grad_output, self, indices);
  return grad_input;
}
} // namespace AtenIpexTypeXPU
} // namespace at
