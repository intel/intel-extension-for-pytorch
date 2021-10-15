#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
struct ContextLinear final {
  ideep::tensor weight_packed_;
  c10::optional<at::Tensor> bias_;

  ContextLinear() = delete;

  ContextLinear(ideep::tensor&& weight_packed, c10::optional<at::Tensor>&& bias)
      : weight_packed_(std::move(weight_packed)), bias_(std::move(bias)) {}

  ContextLinear(ContextLinear&&) = default;
  ContextLinear& operator=(ContextLinear&&) = default;

  ~ContextLinear() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
