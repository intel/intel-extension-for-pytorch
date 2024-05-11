#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_multihead_self_attention(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    at::Tensor seq_info,
    const double scale_attn,
    int64_t max_positions,
    const c10::optional<at::Tensor>& head_mask /* optional */,
    const c10::optional<at::Tensor>& attention_mask /* optional */,
    c10::optional<bool> add_casual_mask /* optional */);

at::Tensor prepare_4d_causal_attention_mask_forward_cpu(
    at::Tensor& attention_mask,
    at::Tensor& inputs_embeds,
    at::Tensor& past_kv_len,
    at::Tensor& finfo_min,
    int64_t sliding_window);
} // namespace

using masked_multihead_self_attention_kernel_fn =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> (*)(
        at::Tensor& query,
        at::Tensor& key,
        at::Tensor& value,
        at::Tensor& key_cache,
        at::Tensor& value_cache,
        at::Tensor& beam_idx,
        at::Tensor seq_info,
        const double scale_attn,
        int64_t max_positions,
        const c10::optional<at::Tensor>& head_mask /* optional */,
        const c10::optional<at::Tensor>& attention_mask /* optional */,
        c10::optional<bool> add_casual_mask /* optional */);

IPEX_DECLARE_DISPATCH(
    masked_multihead_self_attention_kernel_fn,
    masked_multihead_self_attention_kernel_stub);
using prepare_4d_causal_attention_mask_kernel_fn = at::Tensor (*)(
    at::Tensor& attention_mask,
    at::Tensor& inputs_embeds,
    at::Tensor& past_kv_len,
    at::Tensor& finfo_min,
    int64_t sliding_window);

IPEX_DECLARE_DISPATCH(
    prepare_4d_causal_attention_mask_kernel_fn,
    prepare_4d_causal_attention_mask_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
