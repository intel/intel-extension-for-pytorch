#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <tuple>

#include <core/Runtime.h>
#include <utils/Math.h>
#include <utils/ATDispatch.h>
#include "Pooling.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void max_pool2d_with_indices_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input_,
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
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
    : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(input_.ndimension() == 4, "only support 4 dims on DPCPP device now!");

  /* sizes */
  const auto nbatch = input_.size(-4);
  const auto nInputPlane = input_.size(-3);
  const auto inputHeight = input_.size(-2);
  const auto inputWidth = input_.size(-1);

  const auto outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const auto outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

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
      outputWidth);

  /* get contiguous input */
  Tensor input = input_.contiguous();
  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  auto alg_kind = algorithm::pooling_max;
  auto prop_kind = dnnl::prop_kind::forward_training;
  
  max_pool_out_frame<algorithm::pooling_max>(
      input,
      output,
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
}

Tensor& max_pool2d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
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
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
    : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(input.ndimension() == 4, "only support 4 dims on DPCPP device now!");

  /* get contiguous gradOutput */
  const Tensor gradOutput = gradOutput_.contiguous();

  /* sizes */
  const auto nbatch = input.size(-4);
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);
  const auto outputHeight = gradOutput.size(-2);
  const auto outputWidth = gradOutput.size(-1);

  const auto outputHeight_for_shape_check = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const auto outputWidth_for_shape_check = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
      input,
      gradOutput_,
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
      outputHeight_for_shape_check,
      outputWidth_for_shape_check);

  max_pool_backward_out_frame<algorithm::pooling_max>(
      gradInput,
      gradOutput,
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
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
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
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  auto grad_input = at::empty_like(self, MemoryFormat::Contiguous);
  return at::AtenIpexTypeXPU::max_pool2d_with_indices_backward_out(
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


namespace AtenIpexTypeQuantizedXPU {

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode) {
  Tensor output, indices;
  output = _empty_affine_quantized({0},
    input.options(),
    input.q_scale(),
    input.q_zero_point(),
    MemoryFormat::Contiguous); //Relu fusion?
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
