#include "AddSwish.h"
#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>
#include <iostream>

#include <limits>

namespace torch_ipex {
namespace cpu {
DEFINE_DISPATCH(add_swish_kernel_stub);

// Currently we only support 1D tensor of bias(operand of add).
at::Tensor AddSwish(
    at::Tensor& x,
    at::Tensor& mm_output,
    const at::Tensor& weight,
    const at::Tensor& bias) {
#if defined(DYN_DISP_BUILD)
  return add_swish_kernel_stub(kCPU, x, mm_output, weight, bias);
#else
  return add_swish_kernel_impl(x, mm_output, weight, bias);
#endif
}

} // namespace cpu
} // namespace torch_ipex
