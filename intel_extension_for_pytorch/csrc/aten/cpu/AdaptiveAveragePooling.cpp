#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include "AdaptiveAveragePooling.h"
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(adaptive_avg_pool2d_kernel_stub);
DEFINE_DISPATCH(adaptive_avg_pool2d_backward_kernel_stub);

void adaptive_avg_pool2d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    at::IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  int64_t ndim = input.ndimension();
  for (int64_t i = 1; i < ndim; i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size "
        "for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
      input.sizes());
  TORCH_CHECK(
      input.dtype() == output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `output` but got dtype ",
      output.dtype());

  int64_t channels = input.size(-3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  if (ndim == 3) {
    output.resize_({channels, output_height, output_width});
  } else {
    int64_t nbatch = input.size(0);
    output.resize_(
        {nbatch, channels, output_height, output_width},
        input.suggest_memory_format());
  }

  if (output.numel() == 0) {
    return;
  }

#if defined(DYN_DISP_BUILD)
  adaptive_avg_pool2d_kernel_stub(kCPU, output, input, output_size);
#else
  adaptive_avg_pool2d_kernel_impl(output, input, output_size);
#endif
}

at::Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input) {
  int64_t ndim = grad_output.ndimension();
  for (int64_t i = 1; i < ndim; i++) {
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_avg_pool2d_backward(): Expected grad_output to have "
        "non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got ",
      input.sizes());
  TORCH_CHECK(
      input.dtype() == grad_output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());
  TORCH_CHECK(
      input.dtype() == grad_input.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_input` but got dtype ",
      grad_input.dtype());

  grad_input.resize_(input.sizes(), input.suggest_memory_format());
  grad_input.zero_();

#if defined(DYN_DISP_BUILD)
  adaptive_avg_pool2d_backward_kernel_stub(kCPU, grad_input, grad_output);
#else
  adaptive_avg_pool2d_backward_kernel_impl(grad_input, grad_output);
#endif

  return grad_input;
}

at::Tensor& adaptive_avg_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::Tensor& output) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_out_cpu", std::vector<c10::IValue>({}));

  adaptive_avg_pool2d_out_cpu_template(output, input, output_size);
  return output;
}

at::Tensor adaptive_avg_pool2d_cpu(
    at::Tensor const& input,
    at::IntArrayRef output_size) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_cpu", std::vector<c10::IValue>({}));

  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out_cpu_template(output, input, output_size);
  return output;
}

at::Tensor adaptive_avg_pool2d(
    at::Tensor const& input,
    at::IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");

  if (input.is_mkldnn()) {
    return at::mkldnn_adaptive_avg_pool2d(input, output_size);
  }

  if (!input.is_quantized() && output_size[0] == 1 && output_size[1] == 1) {
// in this case, adaptive pooling is just computing mean over hw
// dimensions, which can be done more efficiently
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
    if (xnnpack::use_global_average_pool(input)) {
      return xnnpack::global_average_pool(input);
    }
#endif

    at::Tensor out = input.mean({-1, -2}, /* keepdim = */ true);
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
      // assert ndim == 4, since ndim = 3 doesn't give channels_last
      const int n = input.size(0);
      const int c = input.size(1);
      out.as_strided_({n, c, 1, 1}, {c, 1, c, c});
    }
    return out;
  } else {
    return _adaptive_avg_pool2d(input, output_size);
  }
}

at::Tensor& adaptive_avg_pool2d_backward_out_cpu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_backward_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_backward_out_cpu",
      std::vector<c10::IValue>({}));

  adaptive_avg_pool2d_backward_out_cpu_template(grad_input, grad_output, input);
  return grad_input;
}

at::Tensor adaptive_avg_pool2d_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_backward_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_backward_cpu",
      std::vector<c10::IValue>({}));

  auto grad_input = at::empty({0}, input.options());
  adaptive_avg_pool2d_backward_out_cpu_template(grad_input, grad_output, input);
  return grad_input;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_adaptive_avg_pool2d"),
      TORCH_FN((&torch_ipex::cpu::adaptive_avg_pool2d_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_adaptive_avg_pool2d_backward"),
      TORCH_FN((&torch_ipex::cpu::adaptive_avg_pool2d_backward_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
