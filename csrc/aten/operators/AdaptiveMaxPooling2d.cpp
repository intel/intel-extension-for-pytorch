#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"

#include <vector>

using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void adaptive_max_pool2d_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool2d_dpcpp(): expected input to have non-empty spatial "
        "dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 4),
      "non-empty 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d: internal error: output_size.size() must be 2");

  int64_t outputHeight = output_size[0];
  int64_t outputWidth = output_size[1];

  Tensor input_ = is_smf_channels_last(input) ? input : input.contiguous();
  int64_t nbatch = input_.size(0);
  int64_t nInputPlane = input_.size(1);
  int64_t inputHeight = input_.size(2);
  int64_t inputWidth = input_.size(3);

  int dH = DPCPP::floor((float)2 * inputHeight / outputHeight) -
      DPCPP::floor((float)inputHeight / outputHeight);
  int dW = DPCPP::floor((float)2 * inputWidth / outputWidth) -
      DPCPP::floor((float)inputWidth / outputWidth);

  int kH = DPCPP::ceil((float)2 * inputHeight / outputHeight) -
      DPCPP::floor((float)inputHeight / outputHeight);
  int kW = DPCPP::ceil((float)2 * inputWidth / outputWidth) -
      DPCPP::floor((float)inputWidth / outputWidth);

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;

  if (is_smf_channels_last(input_)) {
    output.resize_(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        at::MemoryFormat::ChannelsLast);
    indices.resize_(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        at::MemoryFormat::ChannelsLast);
  } else {
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
    indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  }

  ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_max>(
      output,
      indices,
      input_,
      nbatch,
      nInputPlane,
      0,
      inputHeight,
      inputWidth,
      0,
      outputHeight,
      outputWidth,
      0,
      kH,
      kW,
      0,
      dH,
      dW,
      0,
      padH,
      padW);
}

Tensor& adaptive_max_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    const Tensor& indices) {
  TORCH_CHECK(
      input.ndimension() == 4, "only support 4 dims on DPCPP device now!");
  Tensor gradOutput;
  /* resize */
  if (is_smf_channels_last(input)) {
    gradInput.resize_as_(input, at::MemoryFormat::ChannelsLast);
    gradOutput = gradOutput_.contiguous(at::MemoryFormat::ChannelsLast);
  } else {
    gradInput.resize_as_(input);
    gradOutput = gradOutput_.contiguous();
  }

  int64_t nbatch = input.size(0);
  int64_t nPlane = input.size(1);
  int64_t gradInputHeight = input.size(2);
  int64_t gradInputWidth = input.size(3);

  int64_t gradOutputHeight = gradOutput.size(2);
  int64_t gradOutputWidth = gradOutput.size(3);

  int dH = DPCPP::floor((float)2 * gradInputHeight / gradOutputHeight) -
      DPCPP::floor((float)gradInputHeight / gradOutputHeight);
  int dW = DPCPP::floor((float)2 * gradInputWidth / gradOutputWidth) -
      DPCPP::floor((float)gradInputWidth / gradOutputWidth);

  int kH = DPCPP::ceil((float)2 * gradInputHeight / gradOutputHeight) -
      DPCPP::floor((float)gradInputHeight / gradOutputHeight);
  int kW = DPCPP::ceil((float)2 * gradInputWidth / gradOutputWidth) -
      DPCPP::floor((float)gradInputWidth / gradOutputWidth);

  int padH = (dH * (gradOutputHeight - 1) + kH - gradInputHeight) / 2;
  int padW = (dW * (gradOutputWidth - 1) + kW - gradInputWidth) / 2;

  ::xpu::oneDNN::pooling_backward<::xpu::oneDNN::alg::pooling_max>(
      gradInput,
      gradOutput,
      indices,
      nbatch,
      nPlane,
      0,
      gradInputHeight,
      gradInputWidth,
      0,
      gradOutputHeight,
      gradOutputWidth,
      0,
      kH,
      kW,
      0,
      dH,
      dW,
      0,
      padH,
      padW);
  return gradInput;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> adaptive_max_pool2d_out(
    Tensor& out,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef output_size) {
  impl::adaptive_max_pool2d_out_template(out, indices, self, output_size);
  return std::tuple<Tensor&, Tensor&>(out, indices);
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d(
    const Tensor& self,
    IntArrayRef output_size) {
  Tensor output = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  TORCH_INTERNAL_ASSERT(output_size.size() == 2);
  return at::AtenIpexTypeXPU::adaptive_max_pool2d_out(
      output, indices, self, output_size);
}

Tensor& adaptive_max_pool2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  impl::adaptive_max_pool2d_backward_out_template(
      grad_input, grad_output, self, indices);
  return grad_input;
}

Tensor adaptive_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  Tensor grad_input = is_smf_channels_last(self)
      ? at::empty_like(self, at::MemoryFormat::ChannelsLast)
      : at::empty_like(self, MemoryFormat::Contiguous);
  return at::AtenIpexTypeXPU::adaptive_max_pool2d_backward_out(
      grad_input, grad_output, self, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
