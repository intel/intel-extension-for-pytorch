#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

#include "AveragePool.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(avg_pool2d_kernel_stub);
DEFINE_DISPATCH(avg_pool2d_backward_kernel_stub);

DEFINE_DISPATCH(avg_pool3d_kernel_stub);
DEFINE_DISPATCH(avg_pool3d_backward_kernel_stub);

at::Tensor avg_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::avg_pool2d_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::avg_pool2d_out_cpu", std::vector<c10::IValue>({}));

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = at::native::pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth = at::native::pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  at::native::pool2d_shape_check(
      input,
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
      memory_format);

  /* resize output */
  at::Tensor output;
  if (input.ndimension() == 3) {
    output =
        at::empty({nInputPlane, outputHeight, outputWidth}, input.options());
  } else {
    output = at::empty(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format));
  }

  /*
  pointer to avg_pool2d_kernel_impl(
      output,
      input,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      count_include_pad,
      divisor_override);
  */
  avg_pool2d_kernel_stub(
      kCPU,
      output,
      input,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      count_include_pad,
      divisor_override);

  return output;
}

at::Tensor avg_pool2d_backward_out_cpu(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::avg_pool2d_backward_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::avg_pool2d_backward_out_cpu", std::vector<c10::IValue>({}));

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // clang-format off
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);
  // clang-format on

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3); // number of channels (or colors)
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputWidth = at::native::pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight = at::native::pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  at::native::avg_pool2d_backward_shape_check(
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

  /* resize output */
  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor gradInput =
      at::empty(input.sizes(), input.options().memory_format(memory_format))
          .zero_();

  TORCH_CHECK(
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `gradOutput` but got dtype ",
      gradOutput.dtype());

  /*
  pointer to avg_pool2d_backward_kernel_impl(
      gradInput,
      gradOutput,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      count_include_pad,
      divisor_override);
  */
  avg_pool2d_backward_kernel_stub(
      kCPU,
      gradInput,
      gradOutput,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      count_include_pad,
      divisor_override);

  return gradInput;
}

at::Tensor avg_pool3d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::avg_pool3d_out_cpu\n");
#endif

  IPEX_RECORD_FUNCTION(
      "torch_ipex::avg_pool3d_out_cpu", std::vector<c10::IValue>({}));

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");

  // clang-format off
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);
  // clang-format on

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    TORCH_CHECK(
        input.ndimension() == 5,
        "non-empty 5D (batch mode) tensor expected for input with channels_last_3d layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 4 || input.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast3d, Contiguous");
  }

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.size(0);
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);
  const int64_t otime = at::native::pooling_output_shape<int64_t>(
      itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight = at::native::pooling_output_shape<int64_t>(
      iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = at::native::pooling_output_shape<int64_t>(
      iwidth, kW, padW, dW, 1, ceil_mode);

  at::native::pool3d_shape_check(
      input,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      1,
      1,
      1,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "avg_pool3d()",
      /*check_input_size=*/true);

  /* resize output */
  at::Tensor output;
  if (input.ndimension() == 4) {
    output = at::empty({nslices, otime, oheight, owidth}, input.options());
  } else {
    output = at::empty(
        {nbatch, nslices, otime, oheight, owidth},
        input.options().memory_format(memory_format));
  }

  /*
  pointer to avg_pool3d_kernel_impl(
      output,
      input,
      kW,
      kH,
      kT,
      dW,
      dH,
      dT,
      padW,
      padH,
      padT,
      count_include_pad,
      divisor_override);
  */
  avg_pool3d_kernel_stub(
      kCPU,
      output,
      input,
      kW,
      kH,
      kT,
      dW,
      dH,
      dT,
      padW,
      padH,
      padT,
      count_include_pad,
      divisor_override);

  return output;
}

at::Tensor avg_pool3d_backward_out_cpu(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::avg_pool3d_backward_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::avg_pool3d_backward_out_cpu", std::vector<c10::IValue>({}));

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");

  // clang-format off
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);
  // clang-format on

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    TORCH_CHECK(
        input.ndimension() == 5,
        "non-empty 5D (batch mode) tensor expected for input with channels_last_3d layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 4 || input.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast3d, Contiguous");
  }

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);
  /* XXX shape check behavior from TH */
  const int64_t otime_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          iwidth, kW, padW, dW, 1, ceil_mode);

  at::native::avg_pool3d_backward_shape_check(
      input,
      gradOutput,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      itime,
      iheight,
      iwidth,
      otime_for_shape_check,
      oheight_for_shape_check,
      owidth_for_shape_check,
      "avg_pool3d_backward()");

  /* resize output */
  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor gradInput =
      at::empty(input.sizes(), input.options().memory_format(memory_format))
          .zero_();

  TORCH_CHECK(
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `gradOutput` but got dtype ",
      gradOutput.dtype());

  /*
  pointer to avg_pool3d_backward_kernel_impl(
      gradInput,
      gradOutput,
      kW,
      kH,
      kT,
      dW,
      dH,
      dT,
      padW,
      padH,
      padT,
      count_include_pad,
      divisor_override);
  */
  avg_pool3d_backward_kernel_stub(
      kCPU,
      gradInput,
      gradOutput,
      kW,
      kH,
      kT,
      dW,
      dH,
      dT,
      padW,
      padH,
      padT,
      count_include_pad,
      divisor_override);

  return gradInput;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::avg_pool2d"),
      TORCH_FN((&torch_ipex::cpu::avg_pool2d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::avg_pool2d_backward"),
      TORCH_FN((&torch_ipex::cpu::avg_pool2d_backward_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::avg_pool3d"),
      TORCH_FN((&torch_ipex::cpu::avg_pool3d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::avg_pool3d_backward"),
      TORCH_FN((&torch_ipex::cpu::avg_pool3d_backward_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex