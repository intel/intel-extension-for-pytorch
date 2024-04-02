#include "RMSNorm.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(rmsnorm_kernel_stub);

at::Tensor dil_RMSNorm(
    const at::Tensor& input,
    const at::Tensor& b,
    double eps) {
  RECORD_FUNCTION("ipex::rmsnorm", c10::ArrayRef<c10::IValue>({}));

  return rmsnorm_kernel_stub(kCPU, input, b, eps);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("rmsnorm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.impl("rmsnorm", c10::DispatchKey::CPU, torch_ipex::cpu::dil_RMSNorm);
}
} // namespace