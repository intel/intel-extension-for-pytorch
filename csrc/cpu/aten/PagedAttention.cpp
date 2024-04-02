#include "PagedAttention.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(single_query_cached_kv_attention_kernel_stub);
IPEX_DEFINE_DISPATCH(reshape_and_cache_kernel_stub);

/*
 *Caculate the masked multihead attention for decoder layer in decoder only
 */
void single_query_cached_kv_attention_forward_cpu(
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
    const c10::optional<at::Tensor>& alibi_slopes) {
  return single_query_cached_kv_attention_kernel_stub(
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
      alibi_slopes);
}

void reshape_and_cache_cpu(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping) {
  return reshape_and_cache_kernel_stub(
      kCPU, key, value, key_cache, value_cache, slot_mapping);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "single_query_cached_kv_attention(Tensor (a!)out, Tensor (a!)query, Tensor (a!)key_cache, Tensor (a!)value_cache,\
       Tensor(a!) head_mapping, float scale, Tensor(a!) block_tables, Tensor(a!) context_lens, int block_size, int max_context_len,\
       Tensor? alibi_slopes)-> ()");
  m.impl(
      "single_query_cached_kv_attention",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::single_query_cached_kv_attention_forward_cpu);
  m.def(
      "reshape_and_cache(Tensor (a!)key, Tensor (a!)value, Tensor (a!)key_cache, Tensor (a!)value_cache, Tensor(a!) slot_mapping)-> ()");
  m.impl(
      "reshape_and_cache",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::reshape_and_cache_cpu);
}
} // namespace
