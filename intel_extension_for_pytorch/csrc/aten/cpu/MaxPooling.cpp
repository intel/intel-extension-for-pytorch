#include "MaxPooling.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(max_pool2d_kernel_stub);
DEFINE_DISPATCH(max_pool2d_backward_kernel_stub);

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::max_pool2d_with_indices_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::max_pool2d_with_indices_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")

  // clang-format off
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);
  // clang-format on

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

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor "
        "expected for input with "
        "channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = at::native::pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = at::native::pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  at::native::pool2d_shape_check(
      input,
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

  /* resize output and indices */
  at::Tensor output, indices;
  if (input.ndimension() == 3) {
    output = at::empty(
        {nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format));
    /* indices will contain the locations for each output point */
    indices = at::empty(
        {nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format).dtype(at::kLong));
  } else {
    output = at::empty(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format));
    /* indices will contain the locations for each output point */
    indices = at::empty(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format).dtype(at::kLong));
  }

#if defined(DYN_DISP_BUILD)
  max_pool2d_kernel_stub(
      kCPU,
      output,
      indices,
      input,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      dilationW,
      dilationH);
#else
  max_pool2d_kernel_impl(
      output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
#endif

  return std::make_tuple(output, indices);
}

at::Tensor max_pool2d_with_indices_backward_out_cpu(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::max_pool2d_with_indices_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::max_pool2d_with_indices_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")

  // clang-format off
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);
  // clang-format on

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
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `gradOutput` but got dtype ",
      gradOutput.dtype());

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor "
        "expected for input with "
        "channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          inputWidth, kW, padW, dW, dilationW, ceil_mode);

  at::native::max_pool2d_backward_shape_check(
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
      outputHeight_for_shape_check,
      outputWidth_for_shape_check,
      memory_format);

  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor gradInput =
      at::empty(input.sizes(), input.options().memory_format(memory_format))
          .zero_();

#if defined(DYN_DISP_BUILD)
  max_pool2d_backward_kernel_stub(kCPU, gradInput, gradOutput, indices);
#else
  max_pool2d_backward_kernel_impl(gradInput, gradOutput, indices);
#endif

  return gradInput;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::max_pool2d_with_indices"),
      TORCH_FN((&torch_ipex::cpu::max_pool2d_with_indices_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::max_pool2d_with_indices_backward"),
      TORCH_FN((&torch_ipex::cpu::max_pool2d_with_indices_backward_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
