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

void adaptive_avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_average_pool2d_dpcpp(): expected input to have non-empty spatial "
        "dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_average_pool2d: internal error: output_size.size() must be 2");

  auto outputWidth = output_size[1];
  auto outputHeight = output_size[0];

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);

  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;

  /* PyTorch support two cases of AdaptiveAvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the suggest_memory_format can only be Contiguous or ChannelsLast1D
     (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
     H, W) case. Then the suggest_memory_format can only be Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor input_;
  if (input.ndimension() == 3) {
    input_ = input.contiguous();
    output.resize_({nInputPlane, outputHeight, outputWidth});
  } else {
    auto smf = input.suggest_memory_format();
    input_ = input.contiguous(smf);
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
  }

  ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      output,
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

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input) {
  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  auto outputHeight = gradOutput.size(-2);
  auto outputWidth = gradOutput.size(-1);

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);

  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;

  auto alg_kind = algorithm::pooling_avg_exclude_padding;

  ::xpu::oneDNN::pooling_backward<
      ::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      gradInput,
      gradOutput,
      input,
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

} // namespace impl

Tensor& adaptive_avg_pool2d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef output_size) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "adaptive_avg_pool2d_out",
      [&]() {
        impl::adaptive_avg_pool2d_out_template(out, self, output_size);
      });
  return out;
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if (self.is_quantized()) {
    output = at::_empty_affine_quantized(
        {0}, self.options(), self.q_scale(), self.q_zero_point());
  } else {
    output = at::empty({0}, self.options());
  }

  return at::AtenIpexTypeXPU::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if (self.is_quantized()) {
    output = at::_empty_affine_quantized(
        {0}, self.options(), self.q_scale(), self.q_zero_point());
  } else {
    output = at::empty({0}, self.options());
  }

  return at::AtenIpexTypeXPU::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor& adaptive_avg_pool2d_backward_out_dpcpp(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& self_) {
  /* PyTorch support two cases of AdaptiveAvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input.resize_as_(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    grad_input.resize_as_(self_, smf);
  }
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "adaptive_avg_pool2d_backward_out_dpcpp",
      [&]() {
        impl::adaptive_avg_pool2d_backward_out_template(
            grad_input, grad_output, self);
      });
  return grad_input;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output_,
    const Tensor& self_) {
  /* PyTorch support two cases of AdaptiveAvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, grad_input;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input = at::empty_like(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    grad_input = at::empty_like(self_, smf);
  }
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "_adaptive_avg_pool2d_backward",
      [&]() {
        impl::adaptive_avg_pool2d_backward_out_template(
            grad_input, grad_output, self);
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor& adaptive_avg_pool2d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef output_size) {
  at::AtenIpexTypeXPU::impl::adaptive_avg_pool2d_out_template(
      out, self, output_size);
  return out;
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  output = at::_empty_affine_quantized(
      {0},
      self.options(),
      self.q_scale(),
      self.q_zero_point(),
      MemoryFormat::Contiguous);
  at::AtenIpexTypeXPU::impl::adaptive_avg_pool2d_out_template(
      output, self, output_size);
  return output;
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  output = at::_empty_affine_quantized(
      {0},
      self.options(),
      self.q_scale(),
      self.q_zero_point(),
      MemoryFormat::Contiguous);
  at::AtenIpexTypeXPU::impl::adaptive_avg_pool2d_out_template(
      output, self, output_size);
  return output;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
