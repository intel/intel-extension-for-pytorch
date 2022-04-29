#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include <vector>

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  const auto outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const auto outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  /* PyTorch support two cases of AvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the suggest_memory_format can only be Contiguous or ChannelsLast1D
     (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
     H, W) case. Then the suggest_memory_format can only be Contiguous.
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
      1,
      1,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      smf);

  /* resize output/indices */
  if (input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth}, smf);
  } else {
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
  }

  if (count_include_pad) {
    ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_include_padding>(
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
  } else {
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
}

Tensor& avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);
  const auto outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const auto outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  avg_pool2d_backward_shape_check(
      input,
      gradOutput,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  if (count_include_pad) {
    ::xpu::oneDNN::pooling_backward<
        ::xpu::oneDNN::alg::pooling_avg_include_padding>(
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
  } else {
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
  return gradInput;
}

} // namespace impl

Tensor& avg_pool2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool2d operator does not support divisor");
  impl::avg_pool2d_out_template(
      output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad);
  return output;
}

Tensor avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  if (input.is_quantized()) {
    output = at::_empty_affine_quantized(
        {0},
        input.options(),
        input.q_scale(),
        input.q_zero_point(),
        MemoryFormat::Contiguous);
  } else {
    output = at::empty({0}, input.options());
  }

  return at::AtenIpexTypeXPU::avg_pool2d_out(
      output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

Tensor& avg_pool2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool2d operator does not support divisor");

  /* PyTorch support two cases of AvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the suggest_memory_format can only be Contiguous or ChannelsLast1D
     (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
     H, W) case. Then the suggest_memory_format can only be Contiguous.
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
    grad_input.resize_as_(self, smf);
  }

  impl::avg_pool2d_backward_out_template(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad);
  return grad_input;
}

Tensor avg_pool2d_backward(
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool2d operator does not support divisor");

  /* PyTorch support two cases of AvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the suggest_memory_format can only be Contiguous or ChannelsLast1D
     (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
     H, W) case. Then the suggest_memory_format can only be Contiguous.
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
    grad_input = at::empty_like(self, smf);
  }

  impl::avg_pool2d_backward_out_template(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad);
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor& avg_pool2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool2d operator does not support divisor");
  at::AtenIpexTypeXPU::impl::avg_pool2d_out_template(
      output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad);
  return output;
}

Tensor avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  output = _empty_affine_quantized(
      {0},
      input.options(),
      input.q_scale(),
      input.q_zero_point(),
      MemoryFormat::Contiguous);

  return at::AtenIpexTypeXPU::avg_pool2d_out(
      output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
