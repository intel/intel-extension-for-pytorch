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
}

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

} // namespace cpu
} // namespace torch_ipex
