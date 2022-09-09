#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinear final {
  ideep::tensor::desc original_desc_;
  ideep::tensor weight_packed_;
  // at_weight will share same memory with weight_packed_
  // at_weight is used for autograd and optimizer update
  at::Tensor at_weight_;
  c10::optional<at::Tensor> bias_;

  ContextLinear() = delete;

  ContextLinear(
      ideep::tensor::desc&& original_desc,
      ideep::tensor&& weight_packed,
      at::Tensor&& at_weight,
      c10::optional<at::Tensor>&& bias)
      : original_desc_(std::move(original_desc)),
        weight_packed_(std::move(weight_packed)),
        at_weight_(std::move(at_weight)),
        bias_(std::move(bias)) {}

  ContextLinear(ContextLinear&&) = default;
  ContextLinear& operator=(ContextLinear&&) = default;

  ~ContextLinear() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
