#include "Decode.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(decode_attention_kernel_stub);
IPEX_DEFINE_DISPATCH(decode_attention_opt_kernel_stub);

// query: [bs, cur_len, num_heads, head_size]
// output: [bs, num_heads, cur_len, head_size_v]
// kv_cache: [max_positions, beam_batch, kv_num_heads, head_size]
// beam_idx: [bs, offset+1]
// attn_logits: [bs, num_heads, num_kv_splits, head_size_v + 1]
at::Tensor decode_attention_forward_cpu(
    at::Tensor& query,
    at::Tensor& output,
    at::Tensor& kv_cache,
    at::Tensor& beam_idx,
    at::Tensor& attn_logits,
    double scaling,
    double logit_cap,
    int64_t offset) {
  return decode_attention_kernel_stub(
      kCPU,
      query,
      output,
      kv_cache,
      beam_idx,
      attn_logits,
      scaling,
      logit_cap,
      offset);
}



// query: [bs, cur_len, num_heads, head_size]
// output: [bs, num_heads, cur_len, head_size_v]
// kv_cache: [max_positions, beam_batch, kv_num_heads, head_size]
// attn_logits: [bs, num_heads, num_kv_splits, head_size_v + 1]
at::Tensor decode_attention_opt_forward_cpu(
  at::Tensor& query,
  at::Tensor& output,
  at::Tensor& kv_cache,
  at::Tensor& attn_logits,
  double scaling,
  double logit_cap,
  int64_t offset) {
return decode_attention_opt_kernel_stub(
    kCPU,
    query,
    output,
    kv_cache,
    attn_logits,
    scaling,
    logit_cap,
    offset);
}
} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "decode_attention(Tensor query, Tensor output, Tensor kv_cache, Tensor beam_idx, \
       Tensor attn_logits, float scaling, float logit_cap, int offset) \
       -> (Tensor)");
  m.impl(
      "decode_attention",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::decode_attention_forward_cpu);
  m.def(
      "decode_attention_opt(Tensor query, Tensor output, Tensor kv_cache, \
        Tensor attn_logits, float scaling, float logit_cap, int offset) \
        -> (Tensor)");
  m.impl(
      "decode_attention_opt",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::decode_attention_opt_forward_cpu);
}
} // namespace
