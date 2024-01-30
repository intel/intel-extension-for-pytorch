#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

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
    at::Tensor&);
at::Tensor mixtral_moe_woq(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&);
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
    at::Tensor&);
using mixtral_moe_tpp_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    bool tpp_fallback,
    const at::Tensor& routing_weights,
    at::Tensor& output);
using mixtral_moe_woq_kernel_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    const at::Tensor& routing_weights,
    at::Tensor& output);
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
    at::Tensor& output);
IPEX_DECLARE_DISPATCH(mixtral_moe_tpp_kernel_fn, mixtral_moe_tpp_kernel_stub);
IPEX_DECLARE_DISPATCH(mixtral_moe_woq_kernel_fn, mixtral_moe_woq_kernel_stub);
IPEX_DECLARE_DISPATCH(mixtral_moe_kernel_fn, mixtral_moe_kernel_stub);
} // namespace cpu
} // namespace torch_ipex
