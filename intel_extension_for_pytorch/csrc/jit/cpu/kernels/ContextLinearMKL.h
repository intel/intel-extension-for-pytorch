#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinearMKL final {
  std::vector<int64_t> sgemm_sizes_ = {0, 0, 0};
  at::Tensor mkl_weight_;
  at::Tensor ori_weight_;
  c10::optional<at::Tensor> bias_;

  ContextLinearMKL() = delete;

  ContextLinearMKL(
      std::vector<int64_t>&& sgemm_sizes,
      at::Tensor&& mkl_weight,
      at::Tensor&& ori_weight,
      c10::optional<at::Tensor>&& bias)
      : sgemm_sizes_(std::move(sgemm_sizes)),
        mkl_weight_(std::move(mkl_weight)),
        ori_weight_(std::move(ori_weight)),
        bias_(std::move(bias)) {}

  ContextLinearMKL(ContextLinearMKL&&) = default;
  ContextLinearMKL& operator=(ContextLinearMKL&&) = default;

  ~ContextLinearMKL() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
