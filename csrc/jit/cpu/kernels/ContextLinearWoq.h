#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinearWoq final {
  at::Tensor at_weight_;
  c10::optional<at::Tensor> at_bias_;

  ContextLinearWoq() = delete;

  ContextLinearWoq(at::Tensor&& at_weight, c10::optional<at::Tensor>&& bias)
      : at_weight_(std::move(at_weight)), at_bias_(std::move(bias)) {}

  ContextLinearWoq(ContextLinearWoq&&) = default;
  ContextLinearWoq& operator=(ContextLinearWoq&&) = default;

  ~ContextLinearWoq() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
