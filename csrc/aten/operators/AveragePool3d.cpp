#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"

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
  //  TensorArg output_arg{ output, "output", 1 };
  //  TensorArg input_arg{ input, "input", 2 };
  //
  //  checkAllSameGPU("avg_pool3d_out_sycl", {output_arg, input_arg});

  Tensor input_ = is_smf_channels_last(input) ? input : input.contiguous();

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
      (input_.ndimension() == 4 || input_.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input_");

  /* sizes */
  const int64_t nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
  const int64_t nblock = input_.size(-4);
  const int64_t idepth = input_.size(-3);
  const int64_t iheight = input_.size(-2);
  const int64_t iwidth = input_.size(-1);

  const int64_t outputDepth =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

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

  if (input_.ndimension() == 4) {
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    if (is_smf_channels_last(input_)) {
      output.resize_(
          {nbatch, nblock, outputDepth, outputHeight, outputWidth},
          at::MemoryFormat::ChannelsLast3d);
    } else {
      output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
    }
  }

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
}

Tensor& avg_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  //  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  //  TensorArg gradOutput_arg{ gradOutput, "gradOutput", 2 };
  //  TensorArg input_arg{ input, "input", 3 };
  //
  //  checkAllSameGPU("avg_pool3d_backward_out_sycl",
  //                  {gradInput_arg, gradOutput_arg, input_arg});

  Tensor gradOutput;
  /* resize */
  auto smf = input.suggest_memory_format();
  if (is_smf_channels_last(input)) {
    gradInput.resize_as_(input, smf);
    gradOutput = gradOutput_.contiguous(smf);
  } else {
    gradInput.resize_as_(input);
    gradOutput = gradOutput_.contiguous();
  }

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
    const Tensor& grad_output,
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
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor grad_input;
  auto smf = self.suggest_memory_format();
  if (is_smf_channels_last(self)) {
    grad_input = at::zeros_like(self, smf);
  } else {
    grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  }
  return at::AtenIpexTypeXPU::avg_pool3d_backward_out(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

} // namespace AtenIpexTypeXPU
} // namespace at
