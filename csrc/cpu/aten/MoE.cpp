#include "MoE.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(mixtral_moe_tpp_kernel_stub);
IPEX_DEFINE_DISPATCH(mixtral_moe_woq_kernel_stub);
IPEX_DEFINE_DISPATCH(deepseek_moe_woq_kernel_stub);
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
    at::Tensor& output,
    bool is_distributed) {
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
      output,
      is_distributed);
}

at::Tensor deepseek_moe_tpp(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_mask,
    const std::vector<at::Tensor>& gate_wei,
    const std::vector<at::Tensor>& up_wei,
    const std::vector<at::Tensor>& down_wei,
    bool tpp_fallback,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed) {
  RECORD_FUNCTION("ipex::deepseek_moe_tpp", c10::ArrayRef<c10::IValue>({}));

  int num_experts = gate_wei.size();
  for (auto i = 0; i < num_experts; i++) {
    auto non_zero = expert_mask[i].nonzero();
    if (non_zero.sizes()[0] == 0)
      continue;
    auto idx = non_zero.select(1, 0);
    auto top_x = non_zero.select(1, 1);
    output = mixtral_moe_tpp_kernel_stub(
        kCPU,
        hidden_states,
        top_x,
        idx,
        gate_wei[i],
        up_wei[i],
        down_wei[i],
        tpp_fallback,
        routing_weights,
        output,
        is_distributed);
  }
  return output;
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
    at::Tensor& output,
    bool is_distributed) {
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
      output,
      is_distributed);
}

at::Tensor deepseek_moe(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_mask,
    const std::vector<at::Tensor>& gate_wei,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>& gate_op_ctx,
    const std::vector<at::Tensor>& up_wei,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>& up_op_ctx,
    const std::vector<at::Tensor>& down_wei,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>& down_op_ctx,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed) {
  RECORD_FUNCTION("ipex::deepseek_moe", c10::ArrayRef<c10::IValue>({}));

  int num_experts = gate_wei.size();
  for (auto i = 0; i < num_experts; i++) {
    auto non_zero = expert_mask[i].nonzero();
    if (non_zero.sizes()[0] == 0)
      continue;
    auto idx = non_zero.select(1, 0);
    auto top_x = non_zero.select(1, 1);

    output = mixtral_moe_kernel_stub(
        kCPU,
        hidden_states,
        top_x,
        idx,
        gate_wei[i],
        gate_op_ctx[i]->get_data_handle(),
        up_wei[i],
        up_op_ctx[i]->get_data_handle(),
        down_wei[i],
        down_op_ctx[i]->get_data_handle(),
        true,
        routing_weights,
        output,
        is_distributed);
  }
  return output;
}

at::Tensor deepseek_moe_mkl(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_mask,
    const std::vector<at::Tensor>& gate_wei,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>& gate_op_ctx,
    const std::vector<at::Tensor>& up_wei,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>& up_op_ctx,
    const std::vector<at::Tensor>& down_wei,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>& down_op_ctx,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed) {
  RECORD_FUNCTION("ipex::deepseek_moe_mkl", c10::ArrayRef<c10::IValue>({}));

  int num_experts = gate_wei.size();
  for (auto i = 0; i < num_experts; i++) {
    auto non_zero = expert_mask[i].nonzero();
    if (non_zero.sizes()[0] == 0)
      continue;
    auto idx = non_zero.select(1, 0);
    auto top_x = non_zero.select(1, 1);
    output = mixtral_moe_kernel_stub(
        kCPU,
        hidden_states,
        top_x,
        idx,
        gate_wei[i],
        gate_op_ctx[i]->get_data_handle(),
        up_wei[i],
        up_op_ctx[i]->get_data_handle(),
        down_wei[i],
        down_op_ctx[i]->get_data_handle(),
        false,
        routing_weights,
        output,
        is_distributed);
  }
  return output;
}
at::Tensor mixtral_moe_woq(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed) {
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
      output,
      is_distributed);
}
at::Tensor deepseek_moe_woq(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_mask,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>& gate_ctx,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>& up_ctx,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>& down_ctx,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed) {
  RECORD_FUNCTION("ipex::deepseek_moe_woq", c10::ArrayRef<c10::IValue>({}));

  int num_experts = gate_ctx.size();
  for (auto i = 0; i < num_experts; i++) {
    auto non_zero = expert_mask[i].nonzero();
    if (non_zero.sizes()[0] == 0)
      continue;
    auto idx = non_zero.select(1, 0);
    auto top_x = non_zero.select(1, 1);
    output = mixtral_moe_woq_kernel_stub(
        kCPU,
        hidden_states,
        top_x,
        idx,
        gate_ctx[i]->get_data_handle(),
        up_ctx[i]->get_data_handle(),
        down_ctx[i]->get_data_handle(),
        routing_weights,
        output,
        is_distributed);
  }
  return output;
}
} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "mixtral_moe_tpp(Tensor hidden_states, Tensor top_x, Tensor idx, Tensor gate_wei, \
      Tensor up_wei, Tensor down_wei, bool tpp_fallback, Tensor routing_weights, \
      Tensor output, bool is_distributed) -> Tensor");
  m.impl(
      "mixtral_moe_tpp",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mixtral_moe_tpp);
  m.def(
      "deepseek_moe_tpp(Tensor hidden_states, Tensor expert_mask, Tensor[] gate_wei, \
      Tensor[] up_wei, Tensor[] down_wei, bool tpp_fallback, Tensor routing_weights, \
      Tensor output, bool is_distributed) -> Tensor");
  m.impl(
      "deepseek_moe_tpp",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::deepseek_moe_tpp);
  m.def(
      "mixtral_moe(Tensor hidden_states, Tensor top_x, Tensor idx, Tensor gate_wei, \
      Tensor gate_op_ctx, Tensor up_wei, Tensor up_op_ctx, Tensor down_wei, \
      Tensor down_op_ctx, bool use_dnnl, Tensor routing_weights, Tensor output, bool is_distributed) -> Tensor");
  m.impl("mixtral_moe", c10::DispatchKey::CPU, torch_ipex::cpu::mixtral_moe);
  m.def(
      "deepseek_moe(Tensor hidden_states, Tensor expert_mask, Tensor[] gate_wei, \
      __torch__.torch.classes.ipex_prepack.LinearOpContext[] gate_op_ctx, Tensor[] up_wei, \
      __torch__.torch.classes.ipex_prepack.LinearOpContext[] up_op_ctx, Tensor[] down_wei, \
      __torch__.torch.classes.ipex_prepack.LinearOpContext[] down_op_ctx, Tensor routing_weights, \
      Tensor output, bool is_distributed) -> Tensor");
  m.impl("deepseek_moe", c10::DispatchKey::CPU, torch_ipex::cpu::deepseek_moe);
  m.def(
      "deepseek_moe_mkl(Tensor hidden_states, Tensor expert_mask, Tensor[] gate_wei, \
      __torch__.torch.classes.ipex_prepack.MKLOpContext[] gate_op_ctx, Tensor[] up_wei, \
      __torch__.torch.classes.ipex_prepack.MKLOpContext[] up_op_ctx, \
      Tensor[] down_wei, __torch__.torch.classes.ipex_prepack.MKLOpContext[] down_op_ctx, \
      Tensor routing_weights, Tensor output, bool is_distributed) -> Tensor");
  m.impl(
      "deepseek_moe_mkl",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::deepseek_moe_mkl);
  m.def(
      "mixtral_moe_woq(Tensor hidden_states, Tensor top_x, Tensor idx, Tensor gate_wei, \
      Tensor up_wei, Tensor down_wei, Tensor routing_weights, Tensor output, bool is_distributed) -> Tensor");
  m.impl(
      "mixtral_moe_woq",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mixtral_moe_woq);
  m.def(
      "deepseek_moe_woq(Tensor hidden_states, Tensor expert_mask, \
      __torch__.torch.classes.ipex_prepack.WoqLinearOpContext[] gate_ctx, \
      __torch__.torch.classes.ipex_prepack.WoqLinearOpContext[] up_ctx, \
      __torch__.torch.classes.ipex_prepack.WoqLinearOpContext[] down_ctx, \
      Tensor routing_weights, Tensor output, bool is_distributed) -> Tensor");

  m.impl(
      "deepseek_moe_woq",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::deepseek_moe_woq);
}
} // namespace