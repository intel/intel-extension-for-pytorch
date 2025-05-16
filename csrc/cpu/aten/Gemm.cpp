#include "Gemm.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "tpp/utils.h"
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
  RECORD_FUNCTION("ipex::bmm_forward_cpu", c10::ArrayRef<c10::IValue>({}));
  return bmm_kernel_stub(
      kCPU, out, mat1, mat2, is_vnni, scale, false, false, block_size_n());
}

at::Tensor convert_weight_packed(
    at::Tensor& weight,
    bool use_tuned_block_n = false) {
  RECORD_FUNCTION(
      "ipex::convert_weight_packed", c10::ArrayRef<c10::IValue>({}));
  return convert_weight_packed_kernel_stub(kCPU, weight, use_tuned_block_n);
}

at::Tensor moe_gate_bmm_forward(
    at::Tensor& mat1,
    at::Tensor& mat2,
    bool is_vnni,
    int64_t topk,
    const c10::optional<at::Tensor>& scale) {
  auto out = at::empty({1, mat1.size(0), topk}, mat1.options());
  RECORD_FUNCTION("ipex::moe_gate_bmm_forward", c10::ArrayRef<c10::IValue>({}));
  auto mat_ = mat1.unsqueeze(0);
  bmm_kernel_stub(kCPU, out, mat_, mat2, is_vnni, scale, true, true, 16);
  return out.squeeze(0);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "bmm(Tensor out, Tensor mat1, Tensor mat2, bool is_vnni, \
       Tensor? scale) -> (Tensor)");
  m.impl("bmm", c10::DispatchKey::CPU, torch_ipex::cpu::bmm_forward_cpu);
  m.def(
      "moe_gate_bmm_forward(Tensor mat1, Tensor mat2, bool is_vnni, \
    int topk, Tensor? scale) -> (Tensor)");
  m.impl(
      "moe_gate_bmm_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::moe_gate_bmm_forward);
  m.def(
      "convert_weight_packed(Tensor weight, bool use_tuned_block_n) -> (Tensor)");
  m.impl(
      "convert_weight_packed",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::convert_weight_packed);
}

} // namespace
