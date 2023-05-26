#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinearWoq final {
  at::Tensor at_weight_;
  c10::optional<at::Tensor> at_bias_;
  // The list contains three dtype versions of bias
  // i.e., fp32, fp16, bf16
  // If bias is not present, it contains empty tensors
  std::vector<at::Tensor> bias_list_;

  ContextLinearWoq() = delete;

  ContextLinearWoq(
      at::Tensor&& at_weight,
      c10::optional<at::Tensor>&& bias)
      : at_weight_(std::move(at_weight)),
        at_bias_(std::move(bias)) {
    if (at_bias_.has_value() && at_bias_.value().defined()) {
        auto bias_fp32 = at_bias_.value();
        auto bias_fp16 = bias_fp32.to(c10::kHalf);
        auto bias_bf16 = bias_fp32.to(c10::kBFloat16);
        bias_list_ = {bias_fp32, bias_fp16, bias_bf16};
    } else {
        // bias tensor is empty (undefined). Leave the check to kernel.
        auto bias_empty = at::Tensor();
        bias_list_ = {bias_empty, bias_empty, bias_empty};
    }
  }

  ContextLinearWoq(ContextLinearWoq&&) = default;
  ContextLinearWoq& operator=(ContextLinearWoq&&) = default;

  ~ContextLinearWoq() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
