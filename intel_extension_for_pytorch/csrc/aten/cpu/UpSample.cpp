#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/accumulate.h>

#include <math.h>
#include <vector>

#include "UpSample.h"

#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(upsample_nearest1d_kernel_stub);
DEFINE_DISPATCH(upsample_nearest2d_kernel_stub);
DEFINE_DISPATCH(upsample_nearest3d_kernel_stub);
DEFINE_DISPATCH(upsample_linear1d_kernel_stub);
DEFINE_DISPATCH(upsample_bilinear2d_kernel_stub);

DEFINE_DISPATCH(upsample_trilinear3d_kernel_stub);
DEFINE_DISPATCH(upsample_bicubic2d_kernel_stub);
DEFINE_DISPATCH(upsample_nearest1d_backward_kernel_stub);
DEFINE_DISPATCH(upsample_nearest2d_backward_kernel_stub);
DEFINE_DISPATCH(upsample_nearest3d_backward_kernel_stub);

DEFINE_DISPATCH(upsample_linear1d_backward_kernel_stub);
DEFINE_DISPATCH(upsample_bilinear2d_backward_kernel_stub);
DEFINE_DISPATCH(upsample_trilinear3d_backward_kernel_stub);

at::Tensor upsample_nearest1d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_nearest1d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_nearest1d_out_cpu", std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  at::Tensor output = at::empty(full_output_size, input.options());

#if defined(DYN_DISP_BUILD)
  upsample_nearest1d_kernel_stub(kCPU, output, input, scales);
#else
  upsample_nearest1d_kernel_impl(output, input, scales);
#endif

  return output;
}

at::Tensor upsample_nearest1d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_nearest1d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_nearest1d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_1d_common_check(input_size, output_size);

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  at::Tensor grad_input = at::zeros(input_size, grad_output.options());

#if defined(DYN_DISP_BUILD)
  upsample_nearest1d_backward_kernel_stub(
      kCPU, grad_input, grad_output, scales);
#else
  upsample_nearest1d_backward_kernel_impl(grad_input, grad_output, scales);
#endif

  return grad_input;
}

at::Tensor upsample_nearest2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_nearest2d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_nearest2d_out_cpu", std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  at::Tensor output = at::empty(
      full_output_size,
      input.options().memory_format(input.suggest_memory_format()));

#if defined(DYN_DISP_BUILD)
  upsample_nearest2d_kernel_stub(kCPU, output, input, scales_h, scales_w);
#else
  upsample_nearest2d_kernel_impl(output, input, scales_h, scales_w);
#endif

  return output;
}

at::Tensor upsample_nearest2d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_nearest2d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_nearest2d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());

  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor grad_input = at::empty(
                              input_size,
                              grad_output.options().memory_format(
                                  grad_output.suggest_memory_format()))
                              .zero_();

#if defined(DYN_DISP_BUILD)
  upsample_nearest2d_backward_kernel_stub(
      kCPU, grad_input, grad_output, scales_h, scales_w);
#else
  upsample_nearest2d_backward_kernel_impl(
      grad_input, grad_output, scales_h, scales_w);
#endif

  return grad_input;
}

at::Tensor upsample_nearest3d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_nearest3d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_nearest3d_out_cpu", std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_3d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  at::Tensor output = at::empty(
      full_output_size,
      input.options().memory_format(input.suggest_memory_format()));

#if defined(DYN_DISP_BUILD)
  upsample_nearest3d_kernel_stub(
      kCPU, output, input, scales_d, scales_h, scales_w);
#else
  upsample_nearest3d_kernel_impl(output, input, scales_d, scales_h, scales_w);
#endif

  return output;
}

at::Tensor upsample_nearest3d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_nearest3d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_nearest3d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_3d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ",
      grad_output.dim());

  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  at::Tensor grad_input = at::zeros(input_size, grad_output.options());

#if defined(DYN_DISP_BUILD)
  upsample_nearest3d_backward_kernel_stub(
      kCPU, grad_input, grad_output, scales_d, scales_h, scales_w);
#else
  upsample_nearest3d_backward_kernel_impl(
      grad_input, grad_output, scales_d, scales_h, scales_w);
#endif

  return grad_input;
}

at::Tensor upsample_linear1d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_linear1d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_linear1d_out_cpu", std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  at::Tensor output = at::empty(full_output_size, input.options());

#if defined(DYN_DISP_BUILD)
  upsample_linear1d_kernel_stub(kCPU, output, input, align_corners, scales);
#else
  upsample_linear1d_kernel_impl(output, input, align_corners, scales);
#endif

  return output;
}

at::Tensor upsample_linear1d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_linear1d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_linear1d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_1d_common_check(input_size, output_size);

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  at::Tensor grad_input = at::zeros(input_size, grad_output.options());

#if defined(DYN_DISP_BUILD)
  upsample_linear1d_backward_kernel_stub(
      kCPU, grad_input, grad_output, align_corners, scales);
#else
  upsample_linear1d_backward_kernel_impl(
      grad_input, grad_output, align_corners, scales);
#endif

  return grad_input;
}

at::Tensor upsample_bilinear2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_bilinear2d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_bilinear2d_out_cpu", std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  at::Tensor output = at::empty(
      full_output_size,
      input.options().memory_format(input.suggest_memory_format()));

#if defined(DYN_DISP_BUILD)
  upsample_bilinear2d_kernel_stub(
      kCPU, output, input, align_corners, scales_h, scales_w);
#else
  upsample_bilinear2d_kernel_impl(
      output, input, align_corners, scales_h, scales_w);
#endif

  return output;
}

at::Tensor upsample_bilinear2d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_bilinear2d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_bilinear2d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());

  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor grad_input = at::empty(
                              input_size,
                              grad_output.options().memory_format(
                                  grad_output.suggest_memory_format()))
                              .zero_();

#if defined(DYN_DISP_BUILD)
  upsample_bilinear2d_backward_kernel_stub(
      kCPU, grad_input, grad_output, align_corners, scales_h, scales_w);
#else
  upsample_bilinear2d_backward_kernel_impl(
      grad_input, grad_output, align_corners, scales_h, scales_w);
#endif

  return grad_input;
}

at::Tensor upsample_trilinear3d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_trilinear3d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_trilinear3d_out_cpu", std::vector<c10::IValue>({}));
#endif

  auto full_output_size =
      at::native::upsample_3d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  at::Tensor output = at::empty(
      full_output_size,
      input.options().memory_format(input.suggest_memory_format()));

#if defined(DYN_DISP_BUILD)
  upsample_trilinear3d_kernel_stub(
      kCPU, output, input, align_corners, scales_d, scales_h, scales_w);
#else
  upsample_trilinear3d_kernel_impl(
      output, input, align_corners, scales_d, scales_h, scales_w);
#endif

  return output;
}

at::Tensor upsample_trilinear3d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::upsample_trilinear3d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::upsample_trilinear3d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif
  auto full_output_size =
      at::native::upsample_3d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ",
      grad_output.dim());

  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  at::Tensor grad_input = at::zeros(input_size, grad_output.options());

#if defined(DYN_DISP_BUILD)
  upsample_trilinear3d_backward_kernel_stub(
      kCPU,
      grad_input,
      grad_output,
      align_corners,
      scales_d,
      scales_h,
      scales_w);
#else
  upsample_trilinear3d_backward_kernel_impl(
      grad_input, grad_output, align_corners, scales_d, scales_h, scales_w);
#endif

  return grad_input;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest1d"),
      TORCH_FN((&torch_ipex::cpu::upsample_nearest1d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest1d_backward"),
      TORCH_FN((&torch_ipex::cpu::upsample_nearest1d_backward_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d"),
      TORCH_FN((&torch_ipex::cpu::upsample_nearest2d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d_backward"),
      TORCH_FN((&torch_ipex::cpu::upsample_nearest2d_backward_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest3d"),
      TORCH_FN((&torch_ipex::cpu::upsample_nearest3d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest3d_backward"),
      TORCH_FN((&torch_ipex::cpu::upsample_nearest3d_backward_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_linear1d"),
      TORCH_FN((&torch_ipex::cpu::upsample_linear1d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_linear1d_backward"),
      TORCH_FN((&torch_ipex::cpu::upsample_linear1d_backward_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d"),
      TORCH_FN((&torch_ipex::cpu::upsample_bilinear2d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d_backward"),
      TORCH_FN((&torch_ipex::cpu::upsample_bilinear2d_backward_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_trilinear3d"),
      TORCH_FN((&torch_ipex::cpu::upsample_trilinear3d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_trilinear3d_backward"),
      TORCH_FN((&torch_ipex::cpu::upsample_trilinear3d_backward_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
