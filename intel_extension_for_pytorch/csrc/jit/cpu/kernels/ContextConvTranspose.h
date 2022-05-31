#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {

struct ContextConvTranspose final {
  ideep::tensor::desc original_desc_;
  ideep::tensor weight_packed_;
  // at_weight will share same memory with weight_packed_
  // at_weight is used for autograd and optimizer update
  at::Tensor at_weight_;
  c10::optional<at::Tensor> bias_;
  // paddings_, strided_, dilation_, output_padding_ here are expanded and
  // might different with those stored on ConvTransposeOpContext.
  // For example, aten deconv2d can accept padding = 2, but onednn deconv2d need
  // expand padding to {2, 2}
  std::vector<int64_t> padding_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> input_size_;
  int64_t groups_;
  // The originin weight != weight_packed_.get_dims() since there is a tranpose
  // for weight, We directly store origin_weight_dims_ here to avoid compute it.
  std::vector<int64_t> origin_weight_dims_;
  bool weight_is_channels_last_;

  ContextConvTranspose() = delete;

  ContextConvTranspose(
      ideep::tensor::desc&& original_desc,
      ideep::tensor&& weight_packed,
      at::Tensor&& at_weight,
      c10::optional<at::Tensor>&& bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> output_padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      int64_t groups,
      std::vector<int64_t> input_size,
      std::vector<int64_t> origin_weight_dims,
      bool weight_is_channels_last)
      : original_desc_(std::move(original_desc)),
        weight_packed_(std::move(weight_packed)),
        at_weight_(std::move(at_weight)),
        bias_(std::move(bias)),
        padding_(padding),
        output_padding_(output_padding),
        stride_(stride),
        dilation_(dilation),
        input_size_(input_size),
        groups_(groups),
        origin_weight_dims_(origin_weight_dims),
        weight_is_channels_last_(weight_is_channels_last) {}

  ContextConvTranspose(ContextConvTranspose&&) = default;
  ContextConvTranspose& operator=(ContextConvTranspose&&) = default;

  ~ContextConvTranspose() {}
};

} // namespace detail
} // namespace cpu
} // namespace torch_ipex
