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

void avg_pool3d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of "
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
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of "
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
      "avg_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  /* Applies a 3D average pooling over an input signal composed of
     several input planes. This op only support 4D and 5D input. 4D: Input (C,
     D, H, W),  Output (C, D0, H0, W0) 5D: Input (N, C, D, H, W),  Output (N, C,
     D0, H0, W0)
  */
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t outputDepth =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

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
      1,
      1,
      1,
      idepth,
      iheight,
      iwidth,
      outputDepth,
      outputHeight,
      outputWidth,
      "avg_pool3d_out_template()",
      /*check_input_size=*/true);

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

  std::vector<int64_t> kernel_size_vec = {kD, kH, kW};
  std::vector<int64_t> stride_vec = {dD, dH, dW};
  std::vector<int64_t> padding_vec = {padD, padH, padW};
  if (count_include_pad) {
    ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_include_padding>(
        output,
        input_,
        nbatch,
        nblock,
        idepth,
        iheight,
        iwidth,
        outputDepth,
        outputHeight,
        outputWidth,
        kernel_size_vec,
        stride_vec,
        padding_vec,
        padding_vec);
  } else {
    ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
        output,
        input_,
        nbatch,
        nblock,
        idepth,
        iheight,
        iwidth,
        outputDepth,
        outputHeight,
        outputWidth,
        kernel_size_vec,
        stride_vec,
        padding_vec,
        padding_vec);
  }
}

Tensor& avg_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of "
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
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of "
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
      "avg_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t odepth = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const int64_t odepth_for_shape_check =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t oheight_for_shape_check =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_chape_check =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
      input,
      gradOutput,
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
      idepth,
      iheight,
      iwidth,
      odepth,
      oheight,
      owidth,
      "avg_pool3d_backward_out_template()");

  if (count_include_pad) {
    ::xpu::oneDNN::pooling_backward<
        ::xpu::oneDNN::alg::pooling_avg_include_padding>(
        gradInput,
        gradOutput,
        input,
        nbatch,
        nblock,
        idepth,
        iheight,
        iwidth,
        odepth,
        oheight,
        owidth,
        kD,
        kH,
        kW,
        dD,
        dH,
        dW,
        padD,
        padH,
        padW);
  } else {
    ::xpu::oneDNN::pooling_backward<
        ::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
        gradInput,
        gradOutput,
        input,
        nbatch,
        nblock,
        idepth,
        iheight,
        iwidth,
        odepth,
        oheight,
        owidth,
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
  return gradInput;
}
} // namespace impl

Tensor& avg_pool3d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool3d operator does not support divisor");
  impl::avg_pool3d_out_template(
      out, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  return out;
}

Tensor avg_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::avg_pool3d_out(
      output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

Tensor& avg_pool3d_backward_out(
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
      "dpcpp_avg_pool3d operator does not support divisor");
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

  impl::avg_pool3d_backward_out_template(
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

Tensor avg_pool3d_backward(
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
      "dpcpp_avg_pool3d operator does not support divisor");

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
    grad_input = at::empty_like(self, smf);
  }
  impl::avg_pool3d_backward_out_template(
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
} // namespace at
