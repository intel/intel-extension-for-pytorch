#pragma once
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymFloat.h>

namespace sdp {

// The same definition as PyTorch
// We define here because head file in PyTorch is not exposed
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : c10::SymFloat(query.sym_size(-1)).sqrt();
  return c10::SymFloat(softmax_scale);
}

c10::optional<at::Tensor> convert_boolean_attn_mask(
    const c10::optional<at::Tensor>& attn_mask,
    caffe2::TypeMeta dtype) {
  // Pass through
  if (!attn_mask.has_value()) {
    return c10::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    // TODO Use the max type of the input and output
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

} // namespace sdp