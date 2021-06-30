#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include "comm/ATDispatch.h"
#include <oneDNN/oneDNN.h>

#include <vector>


using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void adaptive_avg_pool3d_out_template(
    Tensor& output,
    Tensor const& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool3d(): expected input to have non-empty spatial "
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

  auto nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  auto nblock = input.size(-4);
  auto inputDepth = input.size(-3);
  auto inputHeight = input.size(-2);
  auto inputWidth = input.size(-1);

  auto outputDepth = output_size[0];
  auto outputHeight = output_size[1];
  auto outputWidth = output_size[2];

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

  Tensor input_ = input.is_contiguous(at::MemoryFormat::ChannelsLast) ||
          input.is_contiguous(at::MemoryFormat::ChannelsLast3d)
      ? input
      : input.contiguous();
  if (input_.ndimension() == 4) {
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    if (input_.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth},
          at::MemoryFormat::ChannelsLast3d);
    } else {
      output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
    }
  }

  ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      output,
      input_,
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

Tensor& adaptive_avg_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  Tensor gradOutput;
  /* resize */
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    gradInput.resize_as_(input, at::MemoryFormat::ChannelsLast);
    gradOutput = gradOutput_.contiguous(at::MemoryFormat::ChannelsLast);
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    gradInput.resize_as_(input, at::MemoryFormat::ChannelsLast3d);
    gradOutput = gradOutput_.contiguous(at::MemoryFormat::ChannelsLast3d);
  } else {
    gradInput.resize_as_(input);
    gradOutput = gradOutput_.contiguous();
  }

  auto nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  auto nblock = input.size(-4);
  auto gradInputDepth = input.size(-3);
  auto gradInputHeight = input.size(-2);
  auto gradInputWidth = input.size(-1);

  auto gradOutputDepth = gradOutput.size(-3);
  auto gradOutputHeight = gradOutput.size(-2);
  auto gradOutputWidth = gradOutput.size(-1);

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

  ::xpu::oneDNN::pooling_backward<::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      gradInput,
      gradOutput,
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

Tensor& adaptive_avg_pool3d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef output_size) {
  impl::adaptive_avg_pool3d_out_template(out, self, output_size);
  return out;
}

Tensor adaptive_avg_pool3d(const Tensor& self, IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::adaptive_avg_pool3d_out(output, self, output_size);
}

Tensor& adaptive_avg_pool3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self) {
  impl::adaptive_avg_pool3d_backward_out_template(grad_input, grad_output, self);
  return grad_input;
}

Tensor adaptive_avg_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  Tensor grad_input;
  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    grad_input = at::zeros_like(self, MemoryFormat::ChannelsLast);
  } else if (self.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    grad_input = at::zeros_like(self, MemoryFormat::ChannelsLast3d);
  } else {
    grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  }
  impl::adaptive_avg_pool3d_backward_out_template(
      grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
