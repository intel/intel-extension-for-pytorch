// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
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

template <typename T>
void fuse_index_mul_index_add(
    at::Tensor& output,
    const at::Tensor& curr_state,
    const at::Tensor& routing_weights,
    const at::Tensor& top_x,
    const at::Tensor& idx) {
  RECORD_FUNCTION(
      "ipex::fuse_index_mul_index_add", c10::ArrayRef<c10::IValue>({}));
  auto topx_s0 = top_x.size(0);
  auto* output_ptr = output.data_ptr<T>();
  auto* curr_state_ptr = curr_state.data_ptr<T>();
  auto* routing_weights_ptr = routing_weights.data_ptr<T>();
  auto* top_x_ptr = top_x.data_ptr<int64_t>();
  auto* idx_ptr = idx.data_ptr<int64_t>();

  int64_t output_stride0 = output.stride(0);
  int64_t output_stride1 = output.stride(1);
  int64_t curr_state_size2 = curr_state.size(2);
  int64_t curr_state_stride1 = curr_state.stride(1);
  int64_t curr_state_stride2 = curr_state.stride(2);
  int64_t routing_weights_stride0 = routing_weights.stride(0);
  int64_t routing_weights_stride1 = routing_weights.stride(1);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < topx_s0; ++i) {
    for (int j = 0; j < curr_state_size2; ++j) {
      int64_t rw_index = top_x_ptr[i] * routing_weights_stride0 +
          idx_ptr[i] * routing_weights_stride1;
      int64_t cs_index = i * curr_state_stride1 + j * curr_state_stride2;
      int64_t output_index = top_x_ptr[i] * output_stride0 + j * output_stride1;
      output_ptr[output_index] +=
          routing_weights_ptr[rw_index] * curr_state_ptr[cs_index];
    }
  }
}

template <>
void fuse_index_mul_index_add<at::BFloat16>(
    at::Tensor& output,
    const at::Tensor& curr_state,
    const at::Tensor& routing_weights,
    const at::Tensor& top_x,
    const at::Tensor& idx) {
  RECORD_FUNCTION(
      "ipex::fuse_index_mul_index_add", c10::ArrayRef<c10::IValue>({}));
  using lpVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  auto topx_s0 = top_x.size(0);
  auto* output_ptr = output.data_ptr<at::BFloat16>();
  auto* curr_state_ptr = curr_state.data_ptr<at::BFloat16>();
  auto* routing_weights_ptr = routing_weights.data_ptr<at::BFloat16>();
  auto* top_x_ptr = top_x.data_ptr<int64_t>();
  auto* idx_ptr = idx.data_ptr<int64_t>();

  int64_t output_stride0 = output.stride(0);
  int64_t output_stride1 = output.stride(1);
  int64_t curr_state_size2 = curr_state.size(2);
  int64_t curr_state_stride1 = curr_state.stride(1);
  int64_t curr_state_stride2 = curr_state.stride(2);
  int64_t routing_weights_stride0 = routing_weights.stride(0);
  int64_t routing_weights_stride1 = routing_weights.stride(1);
#pragma omp parallel for
  for (int i = 0; i < topx_s0; ++i) {
    int64_t rw_index = top_x_ptr[i] * routing_weights_stride0 +
        idx_ptr[i] * routing_weights_stride1;
    auto rw_v = lpVec(static_cast<at::BFloat16>(routing_weights_ptr[rw_index]));
    for (int j = 0; j < curr_state_size2 - (curr_state_size2 % vec_size);
         j += vec_size) {
      int64_t cs_index = i * curr_state_stride1 + j * curr_state_stride2;
      int64_t output_index = top_x_ptr[i] * output_stride0 + j * output_stride1;
      auto cs_v = lpVec::loadu(curr_state_ptr + cs_index);
      auto out_v = lpVec::loadu(output_ptr + output_index);
      fVec rw_v1, rw_v2, cs_v1, cs_v2, out_v1, out_v2;
      std::tie(rw_v1, rw_v2) = at::vec::convert_to_float(rw_v);
      std::tie(cs_v1, cs_v2) = at::vec::convert_to_float(cs_v);
      std::tie(out_v1, out_v2) = at::vec::convert_to_float(out_v);
      auto output_v1 = out_v1 + cs_v1 * rw_v1;
      auto output_v2 = out_v2 + cs_v2 * rw_v2;
      at::vec::convert_from_float<at::BFloat16>(output_v1, output_v2)
          .store(output_ptr + output_index);
    }
    for (int j = curr_state_size2 - (curr_state_size2 % vec_size);
         j < curr_state_size2;
         ++j) {
      int64_t cs_index = i * curr_state_stride1 + j * curr_state_stride2;
      int64_t output_index = top_x_ptr[i] * output_stride0 + j * output_stride1;
      output_ptr[output_index] +=
          routing_weights_ptr[rw_index] * curr_state_ptr[cs_index];
    }
  }
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

  if (hidden_states.scalar_type() == at::ScalarType::Float) {
    fuse_index_mul_index_add<float>(
        output, curr_state, routing_weights, top_x, idx);
  } else if (hidden_states.scalar_type() == at::ScalarType::BFloat16) {
    fuse_index_mul_index_add<at::BFloat16>(
        output, curr_state, routing_weights, top_x, idx);
  } else {
    auto routing_w = routing_weights.index({top_x, idx}).unsqueeze(-1);
    curr_state = curr_state * routing_w;
    output.index_add_(
        0, top_x, curr_state.squeeze(0).to(hidden_states.dtype()));
  }

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
  if (hidden_states.scalar_type() == at::ScalarType::Float) {
    fuse_index_mul_index_add<float>(
        output, curr_state, routing_weights, top_x, idx);
  } else if (hidden_states.scalar_type() == at::ScalarType::BFloat16) {
    fuse_index_mul_index_add<at::BFloat16>(
        output, curr_state, routing_weights, top_x, idx);
  } else {
    auto routing_w = routing_weights.index({top_x, idx}).unsqueeze(-1);
    curr_state = curr_state * routing_w;
    output.index_add_(
        0, top_x, curr_state.squeeze(0).to(hidden_states.dtype()));
  }

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
  auto curr_state = hidden_states.index({top_x}).unsqueeze(0);
  curr_state = woq_linear_forward(
      woq_linear_mul_forward(
          curr_state, up_wei, {woq_linear_silu_forward(curr_state, gate_wei)}),
      down_wei);
  if (is_distributed) {
    call_AllReduce(curr_state);
  }
  if (hidden_states.scalar_type() == at::ScalarType::Float) {
    fuse_index_mul_index_add<float>(
        output, curr_state, routing_weights, top_x, idx);
  } else if (hidden_states.scalar_type() == at::ScalarType::BFloat16) {
    fuse_index_mul_index_add<at::BFloat16>(
        output, curr_state, routing_weights, top_x, idx);
  } else {
    auto routing_w = routing_weights.index({top_x, idx}).unsqueeze(-1);
    curr_state = curr_state * routing_w;
    output.index_add_(
        0, top_x, curr_state.squeeze(0).to(hidden_states.dtype()));
  }

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
