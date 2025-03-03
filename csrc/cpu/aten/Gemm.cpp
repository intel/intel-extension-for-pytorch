#include "Gemm.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(bmm_kernel_stub);
IPEX_DEFINE_DISPATCH(convert_weight_packed_kernel_stub);
// mat1 : [B, M, K]
// mat2 : [B, N, K] or [B, OC, IC]
// out  : [B, M, N]
// scale: [] 0-dim tensor for per tensor quant
at::Tensor bmm_forward_cpu(
    at::Tensor& out,
    at::Tensor& mat1,
    at::Tensor& mat2,
    bool is_vnni,
    const c10::optional<at::Tensor>& scale) {
  return bmm_kernel_stub(kCPU, out, mat1, mat2, is_vnni, scale);
}

at::Tensor convert_weight_packed(at::Tensor& weight) {
  return convert_weight_packed_kernel_stub(kCPU, weight);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "bmm(Tensor out, Tensor mat1, Tensor mat2, bool is_vnni, \
       Tensor? scale) -> (Tensor)");
  m.impl("bmm", c10::DispatchKey::CPU, torch_ipex::cpu::bmm_forward_cpu);
  m.def("convert_weight_packed(Tensor weight) -> (Tensor)");
  m.impl(
      "convert_weight_packed",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::convert_weight_packed);
}

} // namespace
