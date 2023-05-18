#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include <vector>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

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

  /* Applies a 3D adaptive max pooling over an input signal composed of
     several input planes. This op only support 4D and 5D input. 4D: Input (C,
     D, H, W),  Output (C, D0, H0, W0) 5D: Input (N, C, D, H, W),  Output (N, C,
     D0, H0, W0)
  */
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 3,
      "adaptive_max_pool3d: internal error: output_size.size() must be 3");

  int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  int64_t nblock = input.size(-4);
  int64_t inputDepth = input.size(-3);
  int64_t inputHeight = input.size(-2);
  int64_t inputWidth = input.size(-1);

  int64_t outputDepth = output_size[0];
  int64_t outputHeight = output_size[1];
  int64_t outputWidth = output_size[2];

  int dD = std::floor((float)2 * inputDepth / outputDepth) -
      std::floor((float)inputDepth / outputDepth);
  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> stride_vec = {dD, dH, dW};

  int kD = std::ceil((float)2 * inputDepth / outputDepth) -
      std::floor((float)inputDepth / outputDepth);
  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> kernel_vec = {kD, kH, kW};

  int padD = (dD * (outputDepth - 1) + kD - inputDepth) / 2;
  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;
  std::vector<int64_t> padding_vec = {padD, padH, padW};

  Tensor input_;

  if (input.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    input_ = input.contiguous();
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
    indices.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = input.suggest_memory_format();
    input_ = contiguous_if_needed(input, smf);
    output.resize_(
        {nbatch, nblock, outputDepth, outputHeight, outputWidth}, smf);
    indices.resize_(
        {nbatch, nblock, outputDepth, outputHeight, outputWidth}, smf);
  }

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0, 0};
  ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_max>(
      output,
      indices,
      input_,
      nbatch,
      nblock,
      inputDepth,
      inputHeight,
      inputWidth,
      outputDepth,
      outputHeight,
      outputWidth,
      stride_vec,
      kernel_vec,
      dilation_vec,
      padding_vec,
      padding_vec);
}

Tensor& adaptive_max_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices) {
  int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  int64_t nblock = input.size(-4);
  int64_t gradInputDepth = input.size(-3);
  int64_t gradInputHeight = input.size(-2);
  int64_t gradInputWidth = input.size(-1);

  int64_t gradOutputDepth = gradOutput.size(-3);
  int64_t gradOutputHeight = gradOutput.size(-2);
  int64_t gradOutputWidth = gradOutput.size(-1);

  int dD = std::floor((float)2 * gradInputDepth / gradOutputDepth) -
      std::floor((float)gradInputDepth / gradOutputDepth);
  int dH = std::floor((float)2 * gradInputHeight / gradOutputHeight) -
      std::floor((float)gradInputHeight / gradOutputHeight);
  int dW = std::floor((float)2 * gradInputWidth / gradOutputWidth) -
      std::floor((float)gradInputWidth / gradOutputWidth);
  std::vector<int64_t> stride_vec = {dD, dH, dW};

  int kD = std::ceil((float)2 * gradInputDepth / gradOutputDepth) -
      std::floor((float)gradInputDepth / gradOutputDepth);
  int kH = std::ceil((float)2 * gradInputHeight / gradOutputHeight) -
      std::floor((float)gradInputHeight / gradOutputHeight);
  int kW = std::ceil((float)2 * gradInputWidth / gradOutputWidth) -
      std::floor((float)gradInputWidth / gradOutputWidth);
  std::vector<int64_t> kernel_vec = {kD, kH, kW};

  int padD = (dD * (gradOutputDepth - 1) + kD - gradInputDepth) / 2;
  int padH = (dH * (gradOutputHeight - 1) + kH - gradInputHeight) / 2;
  int padW = (dW * (gradOutputWidth - 1) + kW - gradInputWidth) / 2;
  std::vector<int64_t> padding_vec = {padD, padH, padW};

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0, 0};
  ::xpu::oneDNN::pooling_backward<::xpu::oneDNN::alg::pooling_max>(
      gradInput,
      gradOutput,
      input,
      indices,
      nbatch,
      nblock,
      gradInputDepth,
      gradInputHeight,
      gradInputWidth,
      gradOutputDepth,
      gradOutputHeight,
      gradOutputWidth,
      stride_vec,
      kernel_vec,
      dilation_vec,
      padding_vec,
      padding_vec);
  return gradInput;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> adaptive_max_pool3d_out(
    const Tensor& self,
    IntArrayRef output_size,
    Tensor& out,
    Tensor& indices) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "adaptive_max_pool3d_out",
      [&]() {
        impl::adaptive_max_pool3d_out_template(out, indices, self, output_size);
      });
  return std::tuple<Tensor&, Tensor&>(out, indices);
}

Tensor& adaptive_max_pool3d_backward_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& indices_,
    Tensor& grad_input) {
  Tensor self, grad_output, indices;
  if (self_.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input.resize_as_(self);
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = self_.suggest_memory_format();
    self = contiguous_if_needed(self_, smf);
    grad_output = contiguous_if_needed(grad_output_, smf);
    indices = contiguous_if_needed(indices_, smf);
    grad_input.resize_as_(self, smf);
  }
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "adaptive_max_pool3d_backward_out",
      [&]() {
        impl::adaptive_max_pool3d_backward_out_template(
            grad_input, grad_output, self, indices);
      });
  return grad_input;
}
} // namespace AtenIpexTypeXPU
} // namespace at
