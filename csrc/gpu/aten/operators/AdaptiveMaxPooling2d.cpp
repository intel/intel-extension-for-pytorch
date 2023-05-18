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
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d: internal error: output_size.size() must be 2");

  int64_t outputHeight = output_size[0];
  int64_t outputWidth = output_size[1];

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
  std::vector<int64_t> kernel_size_vec = {kH, kW};
  std::vector<int64_t> stride_vec = {dH, dW};
  std::vector<int64_t> padding_vec = {padH, padW};
  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};

  /* PyTorch support two cases of AdaptiveMaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0) Kernel (kH, kW)
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
    indices.resize_({nInputPlane, outputHeight, outputWidth});
  } else {
    auto smf = input.suggest_memory_format();
    input_ = contiguous_if_needed(input, smf);
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
    indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
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
      stride_vec,
      kernel_size_vec,
      dilation_vec,
      padding_vec,
      padding_vec);
}

Tensor& adaptive_max_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices) {
  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  int64_t outputHeight = gradOutput.size(-2);
  int64_t outputWidth = gradOutput.size(-1);

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> stride_vec = {dH, dW};

  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> kernel_vec = {kH, kW};

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;
  std::vector<int64_t> padding_vec = {padH, padW};

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};
  ::xpu::oneDNN::pooling_backward<::xpu::oneDNN::alg::pooling_max>(
      gradInput,
      gradOutput,
      input,
      indices,
      nbatch,
      nInputPlane,
      0,
      inputHeight,
      inputWidth,
      0,
      outputHeight,
      outputWidth,
      stride_vec,
      kernel_vec,
      dilation_vec,
      padding_vec,
      padding_vec);
  return gradInput;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> adaptive_max_pool2d_out(
    const Tensor& self,
    IntArrayRef output_size,
    Tensor& out,
    Tensor& indices) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "adaptive_max_pool2d_out",
      [&]() {
        impl::adaptive_max_pool2d_out_template(out, indices, self, output_size);
      });
  return std::tuple<Tensor&, Tensor&>(out, indices);
}

Tensor& adaptive_max_pool2d_backward_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& indices_,
    Tensor& grad_input) {
  /* PyTorch support two cases of AdaptiveMaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, indices;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input.resize_as_(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = contiguous_if_needed(self_, smf);
    grad_output = contiguous_if_needed(grad_output_, smf);
    indices = contiguous_if_needed(indices_, smf);
    grad_input.resize_as_(self, smf);
  }
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "adaptive_max_pool2d_backward_out",
      [&]() {
        impl::adaptive_max_pool2d_backward_out_template(
            grad_input, grad_output, self, indices);
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
