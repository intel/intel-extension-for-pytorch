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
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void max_pool3d_with_indices_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
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
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t inputDepth = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputDepth = pooling_output_shape<int64_t>(
      inputDepth, kD, padD, dD, dilationD, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
      input,
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
      "max_pool3d_with_indices_out_template()",
      /*check_input_size=*/true);

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
    input_ = input.contiguous(smf);
    output.resize_(
        {nbatch, nblock, outputDepth, outputHeight, outputWidth}, smf);
    indices.resize_(
        {nbatch, nblock, outputDepth, outputHeight, outputWidth}, smf);
  }

  std::vector<int64_t> kernel_size_vec = {kD, kH, kW};
  std::vector<int64_t> stride_vec = {dD, dH, dW};
  std::vector<int64_t> padding_vec = {padD, padH, padW};
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
      kernel_size_vec,
      stride_vec,
      padding_vec,
      padding_vec);
}

Tensor& max_pool3d_with_indices_backward_out_template(
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
      gradOutputWidth,
      "max_pool3d_with_indices_backward_out_template()");

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
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices_) {
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
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    indices = indices_.contiguous(smf);
    grad_input.resize_as_(self, smf);
  }
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
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices_) {
  Tensor self, grad_output, indices, grad_input;
  if (self_.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input = at::empty_like(self);
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    indices = indices_.contiguous(smf);
    grad_input = at::empty_like(self, smf);
  }
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

} // namespace AtenIpexTypeXPU
} // namespace at
