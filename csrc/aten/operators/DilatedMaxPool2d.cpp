#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"

#include <tuple>

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void max_pool2d_with_indices_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  const auto outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const auto outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  /* PyTorch support two cases of MaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */

  /* get contiguous input */
  auto smf = input.suggest_memory_format();
  Tensor input_;
  if (input.ndimension() == 3) {
    input_ = input.contiguous();
    smf = at::MemoryFormat::Contiguous;
  } else {
    input_ = input.contiguous(smf);
  }

  pool2d_shape_check(
      input_,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      smf);

  /* resize output/indices */
  if (input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth}, smf);
    indices.resize_({nInputPlane, outputHeight, outputWidth}, smf);
  } else {
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

Tensor& max_pool2d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);
  const auto outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const auto outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  max_pool2d_backward_shape_check(
      input,
      gradOutput,
      indices,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

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

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  impl::max_pool2d_with_indices_out_template(
      output,
      indices,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output, indices;
  output = at::empty({0}, input.options());
  indices = at::empty({0}, input.options().dtype(kLong));

  return at::AtenIpexTypeXPU::max_pool2d_with_indices_out(
      output,
      indices,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}

Tensor& max_pool2d_with_indices_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices_) {
  /* PyTorch support two cases of MaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, indices;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input.resize_as_(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    indices = indices_.contiguous(smf);
    grad_input.resize_as_(self, smf);
  }

  impl::max_pool2d_with_indices_backward_out_template(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return grad_input;
}

Tensor max_pool2d_with_indices_backward(
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices_) {
  /* PyTorch support two cases of MaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, indices, grad_input;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input = at::empty_like(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    indices = indices_.contiguous(smf);
    grad_input = at::empty_like(self, smf);
  }
  impl::max_pool2d_with_indices_backward_out_template(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output, indices;
  output = _empty_affine_quantized(
      {0},
      input.options(),
      input.q_scale(),
      input.q_zero_point(),
      MemoryFormat::Contiguous); // Relu fusion?
  indices = at::empty({0}, input.options().dtype(kLong));

  return at::AtenIpexTypeXPU::max_pool2d_with_indices_out(
      output,
      indices,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
