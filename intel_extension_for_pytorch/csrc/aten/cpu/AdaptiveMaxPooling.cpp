#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "AdaptiveMaxPooling.h"
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(adaptive_max_pool2d_kernel_stub);
DEFINE_DISPATCH(adaptive_max_pool2d_backward_kernel_stub);

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_max_pool2d_out_cpu\n");
#endif

  IPEX_RECORD_FUNCTION(
      "torch_ipex::adaptive_max_pool2d_out_cpu", std::vector<c10::IValue>({}));

  int ndim = input.ndimension();
  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: ",
      input.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool2d(): Expected input to have non-zero size "
        "for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d(): internal error: output_size.size() must be 2");

  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  at::Tensor output, indices;
  if (input.ndimension() == 3) {
    output = at::empty({sizeD, osizeH, osizeW}, input.options());
    /* indices will contain i,j locations for each output point */
    indices =
        at::empty({sizeD, osizeH, osizeW}, input.options().dtype(at::kLong));
  } else {
    output = at::empty(
        {sizeB, sizeD, osizeH, osizeW},
        input.options().memory_format(input.suggest_memory_format()));
    /* indices will contain i,j locations for each output point */
    indices = at::empty(
        {sizeB, sizeD, osizeH, osizeW},
        input.options()
            .memory_format(input.suggest_memory_format())
            .dtype(at::kLong));
  }

  // pointer to adaptive_max_pool2d_kernel_impl(output, indices, input,
  // output_size);
  adaptive_max_pool2d_kernel_stub(kCPU, output, indices, input, output_size);

  return std::make_tuple(output, indices);
}

at::Tensor adaptive_max_pool2d_backward_out_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& indices) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_max_pool2d_backward_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::adaptive_max_pool2d_backward_out_cpu",
      std::vector<c10::IValue>({}));

  int64_t ndim = grad_output.ndimension();
  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "adaptive_max_pooling2d_backward(): Expected 3D or 4D "
      "grad_output, but got: ",
      grad_output.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_max_pooling2d_backward(): Expected grad_output to "
        "have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      input.dtype() == grad_output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());

  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor grad_input =
      at::empty(
          input.sizes(),
          input.options().memory_format(input.suggest_memory_format()))
          .zero_();

  // pointer to adaptive_max_pool2d_backward_kernel_impl(grad_input,
  // grad_output, indices);
  adaptive_max_pool2d_backward_kernel_stub(
      kCPU, grad_input, grad_output, indices);

  return grad_input;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::adaptive_max_pool2d"),
      TORCH_FN((&torch_ipex::cpu::adaptive_max_pool2d_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::adaptive_max_pool2d_backward"),
      TORCH_FN((&torch_ipex::cpu::adaptive_max_pool2d_backward_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex