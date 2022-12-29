#pragma once

#include <ATen/Tensor.h>

#include <ideep.hpp>

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinearMKL final {
  std::vector<int64_t> sgemm_sizes_ = {0, 0, 0};
  at::Tensor at_weight_; // packed at weight
  at::Tensor ori_weight_; // non-packed at weight
  c10::optional<at::Tensor> at_bias_;

  ContextLinearMKL() = delete;

  ContextLinearMKL(
      std::vector<int64_t>&& sgemm_sizes,
      at::Tensor&& mkl_weight,
      at::Tensor&& ori_weight,
      c10::optional<at::Tensor>&& bias)
      : sgemm_sizes_(std::move(sgemm_sizes)),
        at_weight_(std::move(mkl_weight)),
        ori_weight_(std::move(ori_weight)),
        at_bias_(std::move(bias)) {}

  ContextLinearMKL(ContextLinearMKL&&) = default;
  ContextLinearMKL& operator=(ContextLinearMKL&&) = default;

  ~ContextLinearMKL() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
