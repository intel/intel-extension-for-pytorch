#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinearWoq final {
  at::Tensor at_weight_;
  c10::optional<at::Tensor> at_bias_;
  // The list contains three dtype versions of bias, scale and zp
  // i.e., fp32, fp16, bf16
  // If bias is not present, it contains empty tensors
  std::vector<at::Tensor> bias_list_;
  std::vector<at::Tensor> scales_list_;
  std::vector<at::Tensor> zero_points_list_;
  bool is_int4_;
  int64_t lowp_mode_;
  int64_t num_concats_;
  int64_t act_quant_mode_;
  // Original weight shape. Weight may be padded after packing
  c10::optional<std::vector<int64_t>> orig_wei_shape_;

  ContextLinearWoq() = delete;

  ContextLinearWoq(
      at::Tensor&& at_weight,
      at::Tensor&& scales_float,
      at::Tensor&& zero_point_float,
      c10::optional<at::Tensor>&& bias,
      bool is_int4 = false,
      int64_t lowp_mode = 0,
      int64_t num_concats = 1,
      int64_t act_quant_mode = 0,
      c10::optional<std::vector<int64_t>>&& orig_wei_shape = c10::nullopt)
      : at_weight_(std::move(at_weight)),
        at_bias_(std::move(bias)),
        is_int4_(is_int4),
        lowp_mode_(lowp_mode),
        num_concats_(num_concats),
        act_quant_mode_(act_quant_mode),
        orig_wei_shape_(std::move(orig_wei_shape)) {
    // Make three dtype versions of scale, zp and bias
    // There is one more dtype for zp
    auto scales_fp16 = scales_float.to(c10::kHalf);
    auto scales_bf16 = scales_float.to(c10::kBFloat16);
    scales_list_ = {scales_float, scales_fp16, scales_bf16};
    auto zp_fp16 = zero_point_float.to(c10::kHalf);
    auto zp_bf16 = zero_point_float.to(c10::kBFloat16);
    auto zp_int8 = zero_point_float.to(c10::kChar);
    zero_points_list_ = {zero_point_float, zp_fp16, zp_bf16, zp_int8};
    if (at_bias_.has_value() && at_bias_.value().defined()) {
      auto& orig_bias = at_bias_.value();
      auto bias_fp32 = at_bias_.value().to(c10::kFloat);
      auto bias_fp16 = at_bias_.value().to(c10::kHalf);
      auto bias_bf16 = at_bias_.value().to(c10::kBFloat16);
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
