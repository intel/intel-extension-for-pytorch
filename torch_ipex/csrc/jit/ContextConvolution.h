#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {

struct ContextConvolution final {
  ideep::tensor weight_packed_;
  c10::optional<at::Tensor> bias_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  bool weight_is_channels_last_;

  ContextConvolution() = delete;

  ContextConvolution(
      ideep::tensor&& weight_packed,
      c10::optional<at::Tensor>&& bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      int64_t groups,
      bool weight_is_channels_last)
      : weight_packed_(std::move(weight_packed)),
        bias_(std::move(bias)),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups),
        weight_is_channels_last_(weight_is_channels_last) {}

  ContextConvolution(ContextConvolution&&) = default;
  ContextConvolution& operator=(ContextConvolution&&) = default;

  ~ContextConvolution() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
