#include "MaskedMultiHeadAttention.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(masked_multihead_self_attention_kernel_stub);

/*
 *Caculate the masked multihead attention for decoder layer in decoder only
 *model.
 *@param query
 *@param key
 *@param value
 *@param key_cache
 *@param value_cache
 *@param beam_idx
 *@param past_kv_steps
 *@param scale_attn
 *@param max_positions
 *@param head_mask
 *@param attention_mask
 *@param add_casual_mask
 *@return {attn_weights, attn_outs}
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_multihead_self_attention_forward_cpu(
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
    c10::optional<bool> add_casual_mask /* optional */) {
  return masked_multihead_self_attention_kernel_stub(
      kCPU,
      query,
      key,
      value,
      key_cache,
      value_cache,
      beam_idx,
      seq_info,
      scale_attn,
      max_positions,
      head_mask,
      attention_mask,
      add_casual_mask);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "masked_multihead_self_attention(Tensor query, Tensor key, Tensor value, Tensor key_cache, \
       Tensor value_cache, Tensor beam_idx, Tensor seq_info, float scale_attn, int max_positions, \
       Tensor? head_mask, Tensor? attention_mask, bool? add_casual_mask=None)-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl(
      "masked_multihead_self_attention",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::masked_multihead_self_attention_forward_cpu);
}
} // namespace
