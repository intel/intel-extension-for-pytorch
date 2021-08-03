#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"

#include <vector>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void max_pool3d_with_indices_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "max_pool3d: kernel_size must either be a single int, or a tuple "
      "of three ints")
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
      "max_pool3d: stride must either be omitted, a single int, or a tuple of "
      "three ints")
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty()
      ? kH
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "max_pool3d: padding must be either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 3,
      "max_pool3d: dilation must be either a single int, or a tuple of three "
      "ints");
  const int dilationD = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
      ? dilationD
      : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
      ? dilationD
      : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK(
      (input_.ndimension() == 4 || input_.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
  const int64_t nblock = input_.size(-4);
  const int64_t inputDepth = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputDepth = pooling_output_shape<int64_t>(
      inputDepth, kD, padD, dD, dilationD, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
      input_,
      nblock,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW,
      dilationD,
      dilationH,
      dilationW,
      inputDepth,
      inputHeight,
      inputWidth,
      outputDepth,
      outputHeight,
      outputWidth,
      /*check_input_size=*/true);

  /* get contiguous input */
  Tensor input = input_.is_contiguous(at::MemoryFormat::ChannelsLast) ||
          input_.is_contiguous(at::MemoryFormat::ChannelsLast3d)
      ? input_
      : input_.contiguous();

  if (input.ndimension() == 4) {
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
    indices.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      output.resize_(
          {nbatch, nblock, outputDepth, outputHeight, outputWidth},
          at::MemoryFormat::ChannelsLast);
      indices.resize_(
          {nbatch, nblock, outputDepth, outputHeight, outputWidth},
          at::MemoryFormat::ChannelsLast);
    } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      output.resize_(
          {nbatch, nblock, outputDepth, outputHeight, outputWidth},
          at::MemoryFormat::ChannelsLast3d);
      indices.resize_(
          {nbatch, nblock, outputDepth, outputHeight, outputWidth},
          at::MemoryFormat::ChannelsLast3d);
    } else {
      output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
      indices.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
    }
  }

  ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_max>(
      output,
      indices,
      input,
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

Tensor& max_pool3d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
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

  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "max_pool3d: kernel_size must either be a single int, or a tuple of "
      "three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "max_pool3d: stride must either be omitted, a single int, or a tuple of "
      "three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty()
      ? kH
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "max_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 3,
      "max_pool3d: dilation must be either a single int, or a tuple of three "
      "ints");
  const int dilationD = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
      ? dilationD
      : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
      ? dilationD
      : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t gradInputDepth = input.size(-3);
  const int64_t gradInputHeight = input.size(-2);
  const int64_t gradInputWidth = input.size(-1);

  const int64_t gradOutputDepth = gradOutput.size(-3);
  const int64_t gradOutputHeight = gradOutput.size(-2);
  const int64_t gradOutputWidth = gradOutput.size(-1);

  max_pool3d_backward_shape_check(
      input,
      gradOutput,
      indices,
      nblock,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW,
      dilationD,
      dilationH,
      dilationW,
      gradInputDepth,
      gradInputHeight,
      gradInputWidth,
      gradOutputDepth,
      gradOutputHeight,
      gradOutputWidth);

  ::xpu::oneDNN::pooling_backward<::xpu::oneDNN::alg::pooling_max>(
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

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out(
    Tensor& out,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  impl::max_pool3d_with_indices_out_template(
      out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::tuple<Tensor&, Tensor&>(out, indices);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::AtenIpexTypeXPU::max_pool3d_with_indices_out(
      output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor& max_pool3d_with_indices_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  impl::max_pool3d_with_indices_backward_out_template(
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

Tensor max_pool3d_with_indices_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  Tensor grad_input;
  if (self.is_contiguous(MemoryFormat::ChannelsLast)) {
    grad_input = at::zeros_like(self, MemoryFormat::ChannelsLast);
  } else if (self.is_contiguous(MemoryFormat::ChannelsLast3d)) {
    grad_input = at::zeros_like(self, MemoryFormat::ChannelsLast3d);
  } else {
    grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  }
  return at::AtenIpexTypeXPU::max_pool3d_with_indices_backward_out(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
