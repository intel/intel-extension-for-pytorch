#include "MoE.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(mixtral_moe_tpp_kernel_stub);
IPEX_DEFINE_DISPATCH(mixtral_moe_woq_kernel_stub);
IPEX_DEFINE_DISPATCH(mixtral_moe_kernel_stub);

at::Tensor mixtral_moe_tpp(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    bool tpp_fallback,
    const at::Tensor& routing_weights,
    at::Tensor& output) {
  RECORD_FUNCTION("ipex::mixtral_moe_tpp", c10::ArrayRef<c10::IValue>({}));

  if (top_x.sizes()[0] == 0)
    return output;
  return mixtral_moe_tpp_kernel_stub(
      kCPU,
      hidden_states,
      top_x,
      idx,
      gate_wei,
      up_wei,
      down_wei,
      tpp_fallback,
      routing_weights,
      output);
}

at::Tensor mixtral_moe(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& gate_op_ctx,
    const at::Tensor& up_wei,
    const at::Tensor& up_op_ctx,
    const at::Tensor& down_wei,
    const at::Tensor& down_op_ctx,
    bool use_dnnl,
    const at::Tensor& routing_weights,
    at::Tensor& output) {
  RECORD_FUNCTION("ipex::mixtral_moe", c10::ArrayRef<c10::IValue>({}));

  if (top_x.sizes()[0] == 0)
    return output;
  return mixtral_moe_kernel_stub(
      kCPU,
      hidden_states,
      top_x,
      idx,
      gate_wei,
      gate_op_ctx,
      up_wei,
      up_op_ctx,
      down_wei,
      down_op_ctx,
      use_dnnl,
      routing_weights,
      output);
}
at::Tensor mixtral_moe_woq(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    const at::Tensor& routing_weights,
    at::Tensor& output) {
  RECORD_FUNCTION("ipex::mixtral_moe_woq", c10::ArrayRef<c10::IValue>({}));

  if (top_x.sizes()[0] == 0)
    return output;
  return mixtral_moe_woq_kernel_stub(
      kCPU,
      hidden_states,
      top_x,
      idx,
      gate_wei,
      up_wei,
      down_wei,
      routing_weights,
      output);
}
} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "mixtral_moe_tpp(Tensor hidden_states, Tensor top_x, Tensor idx, Tensor gate_wei, \
      Tensor up_wei, Tensor down_wei, bool tpp_fallback, Tensor routing_weights, \
      Tensor output) -> Tensor");
  m.impl(
      "mixtral_moe_tpp",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mixtral_moe_tpp);
  m.def(
      "mixtral_moe(Tensor hidden_states, Tensor top_x, Tensor idx, Tensor gate_wei, \
      Tensor gate_op_ctx, Tensor up_wei, Tensor up_op_ctx, Tensor down_wei, \
      Tensor down_op_ctx, bool use_dnnl, Tensor routing_weights, Tensor output) -> Tensor");
  m.impl("mixtral_moe", c10::DispatchKey::CPU, torch_ipex::cpu::mixtral_moe);
  m.def(
      "mixtral_moe_woq(Tensor hidden_states, Tensor top_x, Tensor idx, Tensor gate_wei, \
      Tensor up_wei, Tensor down_wei, Tensor routing_weights, Tensor output) -> Tensor");
  m.impl(
      "mixtral_moe_woq",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mixtral_moe_woq);
}
} // namespace