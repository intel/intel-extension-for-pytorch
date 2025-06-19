#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor single_query_cached_kv_attention_forward_cpu(
    at::Tensor& out, // [num_seqs, num_heads, head_size]
    at::Tensor& query, // [num_seqs, num_heads, head_size]
    at::Tensor& key_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& value_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& head_mapping, // [num_heads]
    const double scale,
    at::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    at::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes,
    int64_t window_size,
    const double k_scale,
    const double v_scale,
    const double softcap);

std::tuple<at::Tensor, at::Tensor> reshape_and_cache_cpu(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale);

at::Tensor flash_attn_varlen_cpu(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& cu_seqlens_q,
    at::Tensor& cu_seqlens_kv,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    const double softmax_scale,
    bool is_causal,
    at::Tensor& block_table,
    const c10::optional<at::Tensor>& alibi_slopes,
    int64_t window_size_left,
    int64_t window_size_right,
    const std::string_view& kv_cache_dtype,
    const double k_scale,
    const double v_scale,
    const double softcap);

} // namespace

using single_query_cached_kv_attention_fn = void (*)(
    at::Tensor& out, // [num_seqs, num_heads, head_size]
    at::Tensor& query, // [num_seqs, num_heads, head_size]
    at::Tensor& key_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& value_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& head_mapping, // [num_heads]
    const double scale,
    at::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    at::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes,
    int64_t window_size,
    const double k_scale,
    const double v_scale,
    const double softcap);

using reshape_and_cache_fn = void (*)(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale);

using flash_attn_var_len_fn = void (*)(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& cu_seqlens_q,
    at::Tensor& cu_seqlens_kv,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    const double softmax_scale,
    bool is_causal,
    at::Tensor& block_table,
    const c10::optional<at::Tensor>& alibi_slopes,
    int64_t window_size_left,
    int64_t window_size_right,
    const std::string_view& kv_cache_dtype,
    const double k_scale,
    const double v_scale,
    const double softcap);

IPEX_DECLARE_DISPATCH(
    single_query_cached_kv_attention_fn,
    single_query_cached_kv_attention_kernel_stub);
IPEX_DECLARE_DISPATCH(reshape_and_cache_fn, reshape_and_cache_kernel_stub);
IPEX_DECLARE_DISPATCH(flash_attn_var_len_fn, flash_attn_var_len_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
