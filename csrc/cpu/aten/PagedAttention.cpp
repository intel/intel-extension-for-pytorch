#include "PagedAttention.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(single_query_cached_kv_attention_kernel_stub);
IPEX_DEFINE_DISPATCH(reshape_and_cache_kernel_stub);
IPEX_DEFINE_DISPATCH(flash_attn_var_len_kernel_stub);

/*
 *Caculate the masked multihead attention for decoder layer in decoder only
 */
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
    const double softcap) {
  single_query_cached_kv_attention_kernel_stub(
      kCPU,
      out,
      query,
      key_cache,
      value_cache,
      head_mapping,
      scale,
      block_tables,
      context_lens,
      block_size,
      max_context_len,
      alibi_slopes,
      window_size,
      k_scale,
      v_scale,
      softcap);
  return out;
}

std::tuple<at::Tensor, at::Tensor> reshape_and_cache_cpu(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale) {
  reshape_and_cache_kernel_stub(
      kCPU,
      key,
      value,
      key_cache,
      value_cache,
      slot_mapping,
      kv_cache_dtype,
      k_scale,
      v_scale);
  return std::make_tuple(key_cache, value_cache);
}

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
    const double softcap) {
  flash_attn_var_len_kernel_stub(
      kCPU,
      out,
      query,
      key,
      value,
      cu_seqlens_q,
      cu_seqlens_kv,
      max_seqlen_q,
      max_seqlen_kv,
      softmax_scale,
      is_causal,
      block_table,
      alibi_slopes,
      window_size_left,
      window_size_right,
      kv_cache_dtype,
      k_scale,
      v_scale,
      softcap);
  return out;
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER_DISPATCH(
      "single_query_cached_kv_attention",
      torch_ipex::cpu::single_query_cached_kv_attention_forward_cpu,
      c10::DispatchKey::CPU);
  IPEX_OP_REGISTER_DISPATCH(
      "reshape_and_cache",
      torch_ipex::cpu::reshape_and_cache_cpu,
      c10::DispatchKey::CPU);
  IPEX_OP_REGISTER_DISPATCH(
      "flash_attn_varlen_func",
      torch_ipex::cpu::flash_attn_varlen_cpu,
      c10::DispatchKey::CPU);
}
} // namespace
