#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>
#include "Linear.h"

namespace torch_ipex {
namespace cpu {

at::Tensor fused_experts(
    const at::Tensor& hidden_states,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    int64_t woq_weight_dtype,
    int64_t woq_group_size,
    int64_t woq_lowp_mode,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w1_zp,
    const std::optional<at::Tensor>& w1_compensation,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w2_zp,
    const std::optional<at::Tensor>& w2_compensation);

using fused_experts_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    int64_t woq_weight_dtype,
    int64_t woq_group_size,
    int64_t woq_lowp_mode,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w1_zp,
    const std::optional<at::Tensor>& w1_compensation,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w2_zp,
    const std::optional<at::Tensor>& w2_compensation);

at::Tensor fused_mlp(
    const at::Tensor& hidden_states,
    const at::Tensor& w1,
    const at::Tensor& w2,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    int64_t woq_weight_dtype,
    int64_t woq_group_size,
    int64_t woq_lowp_mode,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w1_zp,
    const std::optional<at::Tensor>& w1_compensation,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w2_zp,
    const std::optional<at::Tensor>& w2_compensation);

using fused_mlp_fn = at::Tensor (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& w1,
    const at::Tensor& w2,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    int64_t woq_weight_dtype,
    int64_t woq_group_size,
    int64_t woq_lowp_mode,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w1_zp,
    const std::optional<at::Tensor>& w1_compensation,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w2_zp,
    const std::optional<at::Tensor>& w2_compensation);

IPEX_DECLARE_DISPATCH(fused_experts_fn, fused_experts_impl_stub);
IPEX_DECLARE_DISPATCH(fused_mlp_fn, fused_mlp_impl_stub);
} // namespace cpu
} // namespace torch_ipex
