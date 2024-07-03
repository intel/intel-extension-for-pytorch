// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/Parallel.h>
#include <aten/Linear.h>
#include <aten/LinearMKL.h>
#include <aten/MoE.h>
#include <aten/TPPGEMM.h>
#include <c10/util/Exception.h>
#include <immintrin.h>
#include <torch/csrc/autograd/function.h>
#include <algorithm>
#include "tpp/kernels/TPPGEMMKrnl.h"

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor call_AllReduce(const at::Tensor& self) {
  static auto op_allreduce =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("deepspeed_comm::all_reduce", "")
          .typed<at::Tensor(const at::Tensor& self)>();
  auto ret = op_allreduce.call(self);
  return ret;
}

at::Tensor mixtral_moe_tpp_kernl_impl(
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
  auto curr_state = hidden_states.index({top_x}).unsqueeze(0);
  auto routing_w = routing_weights.index({top_x, idx}).unsqueeze(-1);
  if (tpp_fallback) {
    curr_state = at::linear(
        at::silu(at::linear(curr_state, gate_wei)) *
            at::linear(curr_state, up_wei),
        down_wei);
  } else {
    curr_state = tpp_fused_gate_up_proj_forward_cpu(
        curr_state,
        gate_wei,
        at::empty(0, curr_state.options()),
        up_wei,
        at::empty(0, curr_state.options()),
        c10::nullopt);
    curr_state =
        tpp_linear_nobias_forward_cpu(curr_state, down_wei, c10::nullopt);
  }
  if (is_distributed) {
    call_AllReduce(curr_state);
  }
  curr_state = curr_state * routing_w;
  output.index_add_(0, top_x, curr_state.squeeze(0).to(hidden_states.dtype()));

  return output;
}

at::Tensor mixtral_moe_kernl_impl(
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
  auto curr_state = hidden_states.index({top_x}).unsqueeze(0);
  auto routing_w = routing_weights.index({top_x, idx}).unsqueeze(-1);
  if (use_dnnl) {
    curr_state = ipex_linear(
        at::silu(ipex_linear(
            curr_state, gate_wei, c10::nullopt, gate_op_ctx, c10::nullopt)) *
            ipex_linear(
                curr_state, up_wei, c10::nullopt, up_op_ctx, c10::nullopt),
        down_wei,
        c10::nullopt,
        down_op_ctx,
        c10::nullopt);
  } else {
    curr_state = mkl_sgemm_forward(
        at::silu(mkl_sgemm_forward(
            curr_state, gate_wei, c10::nullopt, gate_op_ctx, c10::nullopt)) *
            mkl_sgemm_forward(
                curr_state, up_wei, c10::nullopt, up_op_ctx, c10::nullopt),
        down_wei,
        c10::nullopt,
        down_op_ctx,
        c10::nullopt);
  }
  if (is_distributed) {
    call_AllReduce(curr_state);
  }
  curr_state = curr_state * routing_w;
  output.index_add_(0, top_x, curr_state.squeeze(0).to(hidden_states.dtype()));

  return output;
}

at::Tensor mixtral_moe_woq_kernl_impl(
    const at::Tensor& hidden_states,
    const at::Tensor& top_x,
    const at::Tensor& idx,
    const at::Tensor& gate_wei,
    const at::Tensor& up_wei,
    const at::Tensor& down_wei,
    const at::Tensor& routing_weights,
    at::Tensor& output,
    bool is_distributed) {
  auto topx_s0 = top_x.size(0);
  auto curr_state = hidden_states.index({top_x}).unsqueeze(0);
  auto routing_w = routing_weights.index({top_x, idx}).unsqueeze(-1);
  auto split_size = 64;
  auto total_splits = std::floor((topx_s0 + split_size - 1) / split_size);
  std::vector<torch::Tensor> res;
  for (auto i = 0; i < total_splits; i++) {
    auto idx_end = (i + 1) * split_size;
    if (idx_end > topx_s0) {
      idx_end = topx_s0;
    }
    auto input_i = curr_state.slice(1, i * split_size, idx_end);
    res.push_back(woq_linear_mul_forward(
        input_i, up_wei, {woq_linear_silu_forward(input_i, gate_wei)}));
  }
  curr_state = woq_linear_forward(at::cat(res, 1), down_wei);

  if (is_distributed) {
    call_AllReduce(curr_state);
  }
  curr_state = curr_state * routing_w;
  output.index_add_(0, top_x, curr_state.squeeze(0).to(hidden_states.dtype()));

  return output;
}
} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    mixtral_moe_tpp_kernel_stub,
    &mixtral_moe_tpp_kernl_impl);
IPEX_REGISTER_DISPATCH(
    mixtral_moe_woq_kernel_stub,
    &mixtral_moe_woq_kernl_impl);
IPEX_REGISTER_DISPATCH(mixtral_moe_kernel_stub, &mixtral_moe_kernl_impl);

} // namespace cpu
} // namespace torch_ipex
