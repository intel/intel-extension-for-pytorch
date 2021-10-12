
#pragma once

#include <ATen/Tensor.h>
#include <torch/custom_class.h>

#include "ContextConvolution.h"
#include "ContextLinear.h"
#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

using SerializationTypeConvolutionPrePack = std::tuple<
    at::Tensor,
    c10::optional<at::Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    int64_t,
    bool,
    bool,
    std::vector<int64_t>>;

class ConvolutionOpContext : public torch::jit::CustomClassHolder {
 protected:
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> input_size_;
  int64_t groups_;
  int64_t output_channel_;
  bool weight_is_channels_last_;
  bool weight_is_packed_;

 public:
  SerializationTypeConvolutionPrePack unpack() {
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        kernel_size_,
        groups_,
        output_channel_,
        weight_is_channels_last_,
        weight_is_packed_,
        input_size_);
  }

  virtual at::Tensor run(
      const at::Tensor& input,
      const ideep::attr_t& attr) = 0;
  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) = 0;
};

class IpexConvolutionOpContext final : public ConvolutionOpContext {
 private:
  detail::ContextConvolution op_context_;

 public:
  IpexConvolutionOpContext(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& dilation,
      std::vector<int64_t>&& kernel_size,
      std::vector<int64_t>&& input_size,
      int64_t groups,
      int64_t output_channel,
      bool weight_is_channels_last,
      bool weight_is_packed,
      detail::ContextConvolution&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    stride_ = std::move(stride);
    padding_ = std::move(padding);
    dilation_ = std::move(dilation);
    kernel_size_ = std::move(kernel_size);
    input_size_ = std::move(input_size);
    groups_ = groups;
    output_channel_ = output_channel;
    weight_is_channels_last_ = weight_is_channels_last;
    weight_is_packed_ = weight_is_packed;
  }

  virtual at::Tensor run(const at::Tensor& input, const ideep::attr_t& attr)
      override;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) override;

  static c10::intrusive_ptr<ConvolutionOpContext> create_context(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& dilation,
      std::vector<int64_t>&& kernel_size,
      int64_t groups,
      int64_t output_channel,
      bool weight_is_channels_last,
      bool weight_is_packed,
      std::vector<int64_t>&& input_size);
};

// linear op
using SerializationTypeLinearPrePack = std::tuple<
    at::Tensor,
    c10::optional<at::Tensor>,
    int64_t,
    int64_t,
    int64_t,
    bool>;

class LinearOpContext : public torch::jit::CustomClassHolder {
 protected:
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> orig_bias_;
  int64_t out_features_;
  int64_t in_features_;
  int64_t batch_size_;
  bool weight_is_packed_;

 public:
  SerializationTypeLinearPrePack unpack() {
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        out_features_,
        in_features_,
        batch_size_,
        weight_is_packed_);
  }

  virtual at::Tensor run(
      const at::Tensor& input,
      const ideep::attr_t& attr) = 0;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) = 0;
};

class IpexLinearOpContext final : public LinearOpContext {
 private:
  detail::ContextLinear op_context_;

 public:
  IpexLinearOpContext(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      int64_t out_features,
      int64_t in_features,
      int64_t batch_size,
      bool weight_is_packed,
      detail::ContextLinear&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    out_features_ = out_features;
    in_features_ = in_features;
    batch_size_ = batch_size;
    weight_is_packed_ = weight_is_packed;
  }

  virtual at::Tensor run(const at::Tensor& input, const ideep::attr_t& attr)
      override;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) override;

  static c10::intrusive_ptr<LinearOpContext> create_context(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      int64_t out_features,
      int64_t in_features,
      int64_t batch_size,
      bool weight_is_packed);
};

} // namespace cpu
} // namespace torch_ipex
