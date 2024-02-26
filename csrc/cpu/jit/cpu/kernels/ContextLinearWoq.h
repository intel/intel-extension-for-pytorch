#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinearWoq final {
  at::Tensor at_weight_;
  std::vector<int64_t> weight_shape_;
  c10::optional<at::Tensor> at_bias_;
  c10::optional<at::Tensor> g_idx_;
  // The list contains three dtype versions of bias, scale and zp
  // i.e., fp32, fp16, bf16
  // If bias is not present, it contains empty tensors
  std::vector<at::Tensor> bias_list_;
  std::vector<at::Tensor> scales_list_;
  std::vector<at::Tensor> zero_points_list_;
  bool is_int4_;
  int64_t group_size_;
  int64_t lowp_mode_;
  int64_t num_concats_;
  int64_t act_quant_mode_;

  ContextLinearWoq() = delete;

  ContextLinearWoq(
      at::Tensor&& at_weight,
      std::vector<int64_t>&& weight_shape,
      at::Tensor&& scales_float,
      at::Tensor&& zero_point_float,
      c10::optional<at::Tensor>&& bias,
      c10::optional<at::Tensor>&& g_idx,
      bool is_int4 = false,
      int64_t group_size = -1,
      int64_t lowp_mode = 0,
      int64_t num_concats = 1,
      int64_t act_quant_mode = 0)
      : at_weight_(std::move(at_weight)),
        weight_shape_(std::move(weight_shape)),
        at_bias_(std::move(bias)),
        g_idx_(std::move(g_idx)),
        is_int4_(is_int4),
        group_size_(group_size),
        lowp_mode_(lowp_mode),
        num_concats_(num_concats),
        act_quant_mode_(act_quant_mode) {
    // Make three dtype versions of scale, zp and bias
    // There is one more dtype for zp
    if (group_size > 0) {
      // Reshape scales/zps for data locality in kernel
      // [N, #block_k] -> [N / block_n, block_n, #block_k]
      // -> [#block_n, #block_k, block_n]
      at::Tensor scales_perm, zp_perm;
      if (at_weight_.dim() == 4) {
        // packed weight in 4d (Nc, Kc, block_k, block_n)
        int64_t block_n = at_weight_.size(-1);
        if (is_int4) {
          block_n *= 2;
        }
        TORCH_CHECK(scales_float.size(0) % block_n == 0);
        std::vector<int64_t> reshape_dim = {
            scales_float.size(0) / block_n, block_n, scales_float.size(1)};
        scales_perm = scales_float.view(reshape_dim)
                          .permute({0, 2, 1})
                          .contiguous()
                          .to(c10::kFloat);
        zp_perm =
            zero_point_float.view(reshape_dim).permute({0, 2, 1}).contiguous();
      } else {
        scales_perm = scales_float.to(c10::kFloat);
        zp_perm = zero_point_float;
      }
      auto scales_fp16 = scales_perm.to(c10::kHalf);
      auto scales_bf16 = scales_perm.to(c10::kBFloat16);
      scales_list_ = {scales_perm, scales_fp16, scales_bf16};
      auto zp_fp16 = zp_perm.to(c10::kHalf);
      auto zp_bf16 = zp_perm.to(c10::kBFloat16);
      auto zp_int8 = zp_perm.to(c10::kChar);
      zero_points_list_ = {zp_perm, zp_fp16, zp_bf16, zp_int8};
    } else {
      auto scales_fp32 = scales_float.to(c10::kFloat);
      auto scales_fp16 = scales_float.to(c10::kHalf);
      auto scales_bf16 = scales_float.to(c10::kBFloat16);
      scales_list_ = {scales_fp32, scales_fp16, scales_bf16};
      auto zp_fp16 = zero_point_float.to(c10::kHalf);
      auto zp_bf16 = zero_point_float.to(c10::kBFloat16);
      auto zp_int8 = zero_point_float.to(c10::kChar);
      zero_points_list_ = {zero_point_float, zp_fp16, zp_bf16, zp_int8};
    }
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
