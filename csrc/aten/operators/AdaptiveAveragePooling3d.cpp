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

  /* Applies a 3D adaptive average pooling over an input signal composed of
     several input planes. This op only support 4D and 5D input. 4D: Input (C,
     D, H, W),  Output (C, D0, H0, W0) 5D: Input (N, C, D, H, W),  Output (N, C,
     D0, H0, W0)
  */
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 3,
      "adaptive_average_pool3d: internal error: output_size.size() must be 3");

  auto nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  auto nblock = input.size(-4);
  auto inputDepth = input.size(-3);
  auto inputHeight = input.size(-2);
  auto inputWidth = input.size(-1);

  auto outputDepth = output_size[0];
  auto outputHeight = output_size[1];
  auto outputWidth = output_size[2];

  int dD = std::floor((float)2 * inputDepth / outputDepth) -
      std::floor((float)inputDepth / outputDepth);
  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);

  int kD = std::ceil((float)2 * inputDepth / outputDepth) -
      std::floor((float)inputDepth / outputDepth);
  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);

  int padD = (dD * (outputDepth - 1) + kD - inputDepth) / 2;
  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;

  Tensor input_;
  if (input.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    input_ = input.contiguous();
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = input.suggest_memory_format();
    input_ = input.contiguous(smf);
    output.resize_(
        {nbatch, nblock, outputDepth, outputHeight, outputWidth}, smf);
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
    const Tensor& gradOutput,
    const Tensor& input) {
  auto nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  auto nblock = input.size(-4);
  auto gradInputDepth = input.size(-3);
  auto gradInputHeight = input.size(-2);
  auto gradInputWidth = input.size(-1);

  auto gradOutputDepth = gradOutput.size(-3);
  auto gradOutputHeight = gradOutput.size(-2);
  auto gradOutputWidth = gradOutput.size(-1);

  int dD = std::floor((float)2 * gradInputDepth / gradOutputDepth) -
      std::floor((float)gradInputDepth / gradOutputDepth);
  int dH = std::floor((float)2 * gradInputHeight / gradOutputHeight) -
      std::floor((float)gradInputHeight / gradOutputHeight);
  int dW = std::floor((float)2 * gradInputWidth / gradOutputWidth) -
      std::floor((float)gradInputWidth / gradOutputWidth);

  int kD = std::ceil((float)2 * gradInputDepth / gradOutputDepth) -
      std::floor((float)gradInputDepth / gradOutputDepth);
  int kH = std::ceil((float)2 * gradInputHeight / gradOutputHeight) -
      std::floor((float)gradInputHeight / gradOutputHeight);
  int kW = std::ceil((float)2 * gradInputWidth / gradOutputWidth) -
      std::floor((float)gradInputWidth / gradOutputWidth);

  int padD = (dD * (gradOutputDepth - 1) + kD - gradInputDepth) / 2;
  int padH = (dH * (gradOutputHeight - 1) + kH - gradInputHeight) / 2;
  int padW = (dW * (gradOutputWidth - 1) + kW - gradInputWidth) / 2;

  ::xpu::oneDNN::pooling_backward<
      ::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      gradInput,
      gradOutput,
      input,
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
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "adaptive_avg_pool3d_out",
      [&]() {
        impl::adaptive_avg_pool3d_out_template(out, self, output_size);
      });
  return out;
}

Tensor adaptive_avg_pool3d(const at::Tensor& input, IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");

  if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1) {
    // in this case, adaptive pooling is just computing mean over hw
    // dimensions, which can be done more efficiently
    Tensor out = input.mean({-1, -2, -3}, /* keepdim = */ true);
    return out;
  } else {
    return at::AtenIpexTypeXPU::_adaptive_avg_pool3d(input, output_size);
  }
}

Tensor _adaptive_avg_pool3d(const Tensor& self, IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::adaptive_avg_pool3d_out(
      output, self, output_size);
}

Tensor& adaptive_avg_pool3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& self_) {
  Tensor self, grad_output;
  if (self_.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input.resize_as_(self);
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    grad_input.resize_as_(self_, smf);
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "adaptive_avg_pool3d_backward_out",
      [&]() {
        impl::adaptive_avg_pool3d_backward_out_template(
            grad_input, grad_output, self);
      });
  return grad_input;
}

Tensor _adaptive_avg_pool3d_backward(
    const Tensor& grad_output_,
    const Tensor& self_) {
  Tensor self, grad_output, grad_input;
  if (self_.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input = at::empty_like(self);
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    grad_input = at::empty_like(self_, smf);
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "_adaptive_avg_pool3d_backward",
      [&]() {
        impl::adaptive_avg_pool3d_backward_out_template(
            grad_input, grad_output, self);
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
