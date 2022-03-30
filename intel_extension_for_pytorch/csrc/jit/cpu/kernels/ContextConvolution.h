#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {

struct ContextConvolution final {
  ideep::tensor::desc original_desc_;
  ideep::tensor weight_packed_;
  ideep::tensor bias_;
  // at_weight/at_bias_ will share same memory with weight_packed_/bias_
  // at_weight/at_bias_ is used for autograd and optimizer update
  at::Tensor at_weight_;
  c10::optional<at::Tensor> at_bias_;
  // paddings_, strided_, dilation_, kernel_size_ here are expanded and
  // might different with those stored on ConvolutionOpContext.
  // For example, aten conv2d can accept padding = 2, but onednn conv2d need
  // expand padding to {2, 2}
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> kernel_size_;
  int64_t groups_;
  bool weight_is_channels_last_;
  ideep::convolution_forward_params conv_params_;
  ideep::convolution_forward::super conv_desc_;

  ContextConvolution() = delete;

  ContextConvolution(
      ideep::tensor::desc&& original_desc,
      ideep::tensor&& weight_packed,
      ideep::tensor&& bias,
      at::Tensor&& at_weight,
      c10::optional<at::Tensor> at_bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      std::vector<int64_t> kernel_size,
      int64_t groups,
      bool weight_is_channels_last,
      ideep::convolution_forward_params conv_params,
      dnnl::convolution_forward conv_desc)
      : original_desc_(std::move(original_desc)),
        weight_packed_(std::move(weight_packed)),
        bias_(std::move(bias)),
        at_weight_(std::move(at_weight)),
        at_bias_(std::move(at_bias)),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        kernel_size_(kernel_size),
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
