#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>
#include "Linear.h"

namespace torch_ipex {
namespace cpu {
at::Tensor mixtral_moe_tpp(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    bool,
    const at::Tensor&,
    at::Tensor&,
    bool);
at::Tensor deepseek_moe_tpp(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    bool,
    const at::Tensor&,
    bool);
at::Tensor mixtral_moe_woq(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    bool);
at::Tensor deepseek_moe_woq(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>&,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>&,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>&,
    const at::Tensor&,
    bool);
at::Tensor mixtral_moe_woq(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    bool);
at::Tensor mixtral_moe(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    bool,
    const at::Tensor&,
    at::Tensor&,
    bool);
at::Tensor deepseek_moe(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>&,
    const std::vector<at::Tensor>&,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>&,
    const std::vector<at::Tensor>&,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>&,
    const at::Tensor&,
    bool);
at::Tensor deepseek_moe_mkl(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>&,
    const std::vector<at::Tensor>&,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>&,
    const std::vector<at::Tensor>&,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>&,
    const at::Tensor&,
    bool);
std::tuple<at::Tensor, at::Tensor> deepseek_moegate(
    const at::Tensor& hidden_states,
    const at::Tensor& scores,
    const at::Tensor& routed_scaling_factor,
    const int64_t n_group,
    const int64_t topk_group,
    const int64_t n_routed_experts,
    const int64_t top_k,
    c10::optional<at::Tensor> e_score_cbias);
using mixtral_moe_tpp_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    bool tpp_fallback,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed);
using deepseek_moe_tpp_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& topk_ids,
    const std::vector<at::Tensor>& gate_wei,
    const std::vector<at::Tensor>& up_wei,
    const std::vector<at::Tensor>& down_wei,
    bool tpp_fallback,
    const at::Tensor& routing_weights,
    bool is_distributed);
using mixtral_moe_woq_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed);
using deepseek_moe_woq_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& topk_ids,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>& gate_ctx,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>& up_ctx,
    const std::vector<c10::intrusive_ptr<WoqLinearOpContext>>& down_ctx,
    const at::Tensor& routing_weights,
    bool is_distributed);
using mixtral_moe_kernel_fn = at::Tensor (*)(
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
    bool is_distributed);
using deepseek_moe_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& topk_ids,
    const std::vector<at::Tensor>& gate_wei,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>& gate_op_ctx,
    const std::vector<at::Tensor>& up_wei,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>& up_op_ctx,
    const std::vector<at::Tensor>& down_wei,
    const std::vector<c10::intrusive_ptr<LinearOpContext>>& down_op_ctx,
    const at::Tensor& routing_weights,
    bool is_distributed);
using deepseek_moe_mkl_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& topk_ids,
    const std::vector<at::Tensor>& gate_wei,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>& gate_op_ctx,
    const std::vector<at::Tensor>& up_wei,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>& up_op_ctx,
    const std::vector<at::Tensor>& down_wei,
    const std::vector<c10::intrusive_ptr<MKLOpContext>>& down_op_ctx,
    const at::Tensor& routing_weights,
    bool is_distributed);
using deepseek_moegate_kernel_fn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& scores,
    const at::Tensor& routed_scaling_factor,
    const int64_t n_group,
    const int64_t topk_group,
    const int64_t n_routed_experts,
    const int64_t top_k,
    c10::optional<at::Tensor> e_score_cbias);
IPEX_DECLARE_DISPATCH(mixtral_moe_tpp_kernel_fn, mixtral_moe_tpp_kernel_stub);
IPEX_DECLARE_DISPATCH(deepseek_moe_tpp_kernel_fn, deepseek_moe_tpp_kernel_stub);
IPEX_DECLARE_DISPATCH(mixtral_moe_woq_kernel_fn, mixtral_moe_woq_kernel_stub);
IPEX_DECLARE_DISPATCH(deepseek_moe_woq_kernel_fn, deepseek_moe_woq_kernel_stub);
IPEX_DECLARE_DISPATCH(mixtral_moe_kernel_fn, mixtral_moe_kernel_stub);
IPEX_DECLARE_DISPATCH(deepseek_moe_kernel_fn, deepseek_moe_kernel_stub);
IPEX_DECLARE_DISPATCH(deepseek_moe_mkl_kernel_fn, deepseek_moe_mkl_kernel_stub);
IPEX_DECLARE_DISPATCH(deepseek_moegate_kernel_fn, deepseek_moegate_kernel_stub);
} // namespace cpu
} // namespace torch_ipex
