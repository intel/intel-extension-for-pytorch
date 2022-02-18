#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {

struct ContextConvolution final {
  ideep::tensor weight_packed_;
  ideep::tensor bias_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  bool weight_is_channels_last_;
  ideep::convolution_forward_params conv_params_;
  ideep::convolution_forward::super conv_desc_;

  ContextConvolution() = delete;

  ContextConvolution(
      ideep::tensor&& weight_packed,
      ideep::tensor&& bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      int64_t groups,
      bool weight_is_channels_last,
      ideep::convolution_forward_params conv_params,
      dnnl::convolution_forward conv_desc)
      : weight_packed_(std::move(weight_packed)),
        bias_(std::move(bias)),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups),
        weight_is_channels_last_(weight_is_channels_last),
        conv_params_(conv_params),
        conv_desc_(conv_desc) {}

  ContextConvolution(ContextConvolution&&) = default;
  ContextConvolution& operator=(ContextConvolution&&) = default;

  ~ContextConvolution() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
