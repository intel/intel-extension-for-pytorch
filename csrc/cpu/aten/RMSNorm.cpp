#include "RMSNorm.h"
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(rmsnorm_kernel_stub);

at::Tensor dil_RMSNorm(
    const at::Tensor& input,
    const at::Tensor& b,
    float eps) {
  RECORD_FUNCTION("dil_RMSNorm", c10::ArrayRef<c10::IValue>({}));

  return rmsnorm_kernel_stub(kCPU, input, b, eps);
}
} // namespace cpu
} // namespace torch_ipex
