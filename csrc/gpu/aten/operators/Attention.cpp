#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_forward(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool need_attn_weights,
    bool is_causal) {
  return at::_scaled_dot_product_attention_math(
      query_, key, value, attn_mask_, dropout_p, need_attn_weights, is_causal);
}

} // namespace AtenIpexTypeXPU
} // namespace at
