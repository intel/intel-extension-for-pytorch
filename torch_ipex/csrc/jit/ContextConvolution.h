#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {

struct ContextConvolution final {
  ideep::tensor weight_packed_;
  c10::optional<at::Tensor> bias_;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  std::array<int64_t, 4> input_size_;
  int64_t groups_;
  bool weight_is_channels_last_;

  ContextConvolution() = delete;

  ContextConvolution(
      ideep::tensor&& weight_packed,
      c10::optional<at::Tensor>&& bias,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      int64_t groups,
      std::array<int64_t, 4> input_size,
      bool weight_is_channels_last)
      : weight_packed_(std::move(weight_packed)),
        bias_(std::move(bias)),
        weight_size_(weight_size),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        input_size_(input_size),
        groups_(groups),
        weight_is_channels_last_(weight_is_channels_last) {}

  ContextConvolution(ContextConvolution&&) = default;
  ContextConvolution& operator=(ContextConvolution&&) = default;

  ~ContextConvolution() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
