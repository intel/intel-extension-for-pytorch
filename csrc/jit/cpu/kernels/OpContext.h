
#pragma once

#include <ATen/Tensor.h>
#include <torch/custom_class.h>

#include <ideep.hpp>
#include "ContextConvTranspose.h"
#include "ContextConvolution.h"
#include "ContextLinear.h"
#include "ContextLinearMKL.h"

namespace torch_ipex {
namespace cpu {

using SerializationTypeConvolutionPrePack = std::tuple<
    at::Tensor,
    c10::optional<at::Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    bool,
    std::vector<int64_t>>;

class ConvolutionOpContext : public torch::jit::CustomClassHolder {
 protected:
  // these origin parameters are used for serialization
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> input_size_;

 public:
  SerializationTypeConvolutionPrePack unpack() {
    auto orig_weight_ = this->to_public(this->get_at_packed_weight());
    auto orig_bias_ = this->get_context().at_bias_;
    auto groups_ = this->get_context().groups_;
    auto weight_is_channels_last_ =
        this->get_context().weight_is_channels_last_;
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        weight_is_channels_last_,
        input_size_);
  }

  virtual at::Tensor run(
      const at::Tensor& input,
      const ideep::attr_t& attr) = 0;
  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) = 0;

  // Runing backward for conv by given grad_output, input and grad_masks.
  // Will using the mkldnn_weight/bias stored in the context
  virtual std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      std::array<bool, 3> output_mask) = 0;

  // Return the n-D ATen weight which sharing same memory with the mkldnn packed
  // weight This n-D ATen weight will be used for autograd and optimizer update
  virtual at::Tensor get_at_packed_weight() = 0;

  // Pack given tensor to same format with mkldnn packed weight
  virtual at::Tensor pack(const at::Tensor& tensor) = 0;

  // Unpack given tensor to same format with original public format for weight
  virtual at::Tensor to_public(const at::Tensor& tensor) = 0;

  std::vector<int64_t> get_stride();

  std::vector<int64_t> get_padding();

  std::vector<int64_t> get_dilation();

  int64_t get_groups();

  virtual detail::ContextConvolution& get_context() = 0;

  virtual at::Tensor get_data_handle() = 0;
};

class IpexConvolutionOpContext final : public ConvolutionOpContext {
 private:
  detail::ContextConvolution op_context_;

 public:
  IpexConvolutionOpContext(
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& dilation,
      std::vector<int64_t>&& input_size,
      detail::ContextConvolution&& op_context)
      : op_context_(std::move(op_context)) {
    stride_ = std::move(stride);
    padding_ = std::move(padding);
    dilation_ = std::move(dilation);
    input_size_ = std::move(input_size);
  }

  virtual at::Tensor run(const at::Tensor& input, const ideep::attr_t& attr)
      override;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) override;

  virtual std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      std::array<bool, 3> output_mask) override;

  virtual at::Tensor get_at_packed_weight() override;

  virtual at::Tensor pack(const at::Tensor& tensor) override;

  virtual at::Tensor to_public(const at::Tensor& tensor) override;

  virtual detail::ContextConvolution& get_context() override;

  virtual at::Tensor get_data_handle() override;

  static c10::intrusive_ptr<ConvolutionOpContext> create_context(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      bool weight_is_channels_last,
      std::vector<int64_t>&& input_size,
      const ideep::attr_t& attr);
};

// linear op
using SerializationTypeLinearPrePack =
    std::tuple<at::Tensor, c10::optional<at::Tensor>, c10::optional<int64_t>>;

class LinearOpContext : public torch::jit::CustomClassHolder {
 protected:
  c10::optional<int64_t> batch_size_;

 public:
  SerializationTypeLinearPrePack unpack() {
    auto orig_weight_ = this->to_public(this->get_at_packed_weight());
    auto orig_bias_ = this->get_context().bias_;
    return std::make_tuple(orig_weight_, orig_bias_, batch_size_);
  }

  virtual at::Tensor get_data_handle() = 0;

  virtual at::Tensor run(
      const at::Tensor& input,
      const ideep::attr_t& attr) = 0;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) = 0;

  // Runing backward for linear by given grad_output, input and grad_masks.
  // Will using the mkldnn_weight stored in the context
  virtual std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      std::array<bool, 3> output_mask) = 0;

  // Return the n-D ATen weight which sharing same memory with the mkldnn packed
  // weight This n-D ATen weight will be used for autograd and optimizer update
  virtual at::Tensor get_at_packed_weight() = 0;

  // Pack given tensor to same format with mkldnn packed weight
  virtual at::Tensor pack(const at::Tensor& tensor) = 0;

  // Unpack given tensor to same format with original public format for weight
  virtual at::Tensor to_public(const at::Tensor& tensor) = 0;

  virtual detail::ContextLinear& get_context() = 0;
};

class IpexLinearOpContext final : public LinearOpContext {
 private:
  detail::ContextLinear op_context_;

 public:
  IpexLinearOpContext(
      c10::optional<int64_t> batch_size,
      detail::ContextLinear&& op_context)
      : op_context_(std::move(op_context)) {
    batch_size_ = batch_size;
  }

  virtual at::Tensor run(const at::Tensor& input, const ideep::attr_t& attr)
      override;

  virtual at::Tensor get_data_handle() override;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) override;

  virtual std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      std::array<bool, 3> output_mask) override;

  virtual at::Tensor get_at_packed_weight() override;

  virtual at::Tensor pack(const at::Tensor& tensor) override;

  virtual at::Tensor to_public(const at::Tensor& tensor) override;

  virtual detail::ContextLinear& get_context() override;

  static c10::intrusive_ptr<LinearOpContext> create_context(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      c10::optional<int64_t> batch_size);
};

using SerializationTypeMKLPrePack =
    std::tuple<at::Tensor, c10::optional<at::Tensor>, c10::optional<int64_t>>;

class MKLOpContext : public torch::jit::CustomClassHolder {
 protected:
  c10::optional<int64_t> batch_size_;

 public:
  SerializationTypeMKLPrePack unpack() {
    auto orig_weight = this->to_public(this->get_at_packed_weight());
    auto orig_bias = this->get_mkl_context().bias_;
    return std::make_tuple(orig_weight, orig_bias, batch_size_);
  }

  virtual at::Tensor get_at_packed_weight() = 0;

  virtual at::Tensor get_data_handle() = 0;

  virtual at::Tensor pack(const at::Tensor& tensor) = 0;

  virtual at::Tensor run(const at::Tensor& input) = 0;

  virtual at::Tensor& run(const at::Tensor& input, at::Tensor& accumu) = 0;

  // Unpack given tensor to same format with original public format for weight
  virtual at::Tensor to_public(const at::Tensor& tensor) = 0;

  virtual int64_t get_out_features() = 0;

  virtual int64_t get_in_features() = 0;

  virtual detail::ContextLinearMKL& get_mkl_context() = 0;

  c10::optional<int64_t> get_batchsize();
};

class IpexLinearMKLOpContext final : public MKLOpContext {
 private:
  detail::ContextLinearMKL op_context_;

 public:
  IpexLinearMKLOpContext(
      c10::optional<int64_t> batch_size,
      detail::ContextLinearMKL&& op_context)
      : op_context_(std::move(op_context)) {
    batch_size_ = batch_size;
  }

  virtual at::Tensor get_at_packed_weight() override;

  virtual at::Tensor get_data_handle() override;

  virtual at::Tensor pack(const at::Tensor& tensor) override;

  virtual at::Tensor run(const at::Tensor& input) override;

  virtual at::Tensor& run(const at::Tensor& input, at::Tensor& accumu) override;

  virtual at::Tensor to_public(const at::Tensor& tensor) override;

  virtual detail::ContextLinearMKL& get_mkl_context() override;

  virtual int64_t get_out_features() override;

  virtual int64_t get_in_features() override;

  static c10::intrusive_ptr<MKLOpContext> create_context(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      c10::optional<int64_t> batch_size);
};

// deconv op
using SerializationTypeConvTransposePrePack = std::tuple<
    at::Tensor,
    c10::optional<at::Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::vector<int64_t>,
    bool,
    std::vector<int64_t>>;

class ConvTransposeOpContext : public torch::jit::CustomClassHolder {
 protected:
  // these origin parameters are used for serialization
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> input_size_;

 public:
  SerializationTypeConvTransposePrePack unpack() {
    auto orig_weight_ = this->to_public(this->get_at_packed_weight());
    auto orig_bias_ = this->get_context().bias_;
    auto groups_ = this->get_context().groups_;
    auto weight_is_channels_last_ =
        this->get_context().weight_is_channels_last_;
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        output_padding_,
        groups_,
        dilation_,
        weight_is_channels_last_,
        input_size_);
  }

  virtual at::Tensor run(
      const at::Tensor& input,
      const ideep::attr_t& attr) = 0;
  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) = 0;

  // Runing backward for conv_transpose by given grad_output, input and
  // grad_masks. Will using the mkldnn_weight stored in the context
  virtual std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      std::array<bool, 3> output_mask) = 0;

  // Return the n-D ATen weight which sharing same memory with the mkldnn packed
  // weight This n-D ATen weight will be used for autograd and optimizer update
  virtual at::Tensor get_at_packed_weight() = 0;

  // Pack given tensor to same format with mkldnn packed weight
  virtual at::Tensor pack(const at::Tensor& tensor) = 0;

  // Unpack given tensor to same format with original public format for weight
  virtual at::Tensor to_public(const at::Tensor& tensor) = 0;

  // query best weight format by given input size, and re-pack the mkldnn weight
  // to newly queried format
  virtual void may_repack(std::vector<int64_t> input_size) = 0;

  virtual at::Tensor get_data_handle() = 0;

  virtual detail::ContextConvTranspose& get_context() = 0;
};

class IpexConvTransposeOpContext final : public ConvTransposeOpContext {
 private:
  detail::ContextConvTranspose op_context_;

 public:
  IpexConvTransposeOpContext(
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& dilation,
      std::vector<int64_t>&& input_size,
      detail::ContextConvTranspose&& op_context)
      : op_context_(std::move(op_context)) {
    stride_ = std::move(stride);
    padding_ = std::move(padding);
    output_padding_ = std::move(output_padding);
    dilation_ = std::move(dilation);
    input_size_ = std::move(input_size);
  }

  virtual at::Tensor run(const at::Tensor& input, const ideep::attr_t& attr)
      override;

  virtual at::Tensor& run(
      const at::Tensor& input,
      at::Tensor& accumu,
      const ideep::attr_t& attr) override;

  virtual std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
      const at::Tensor& input,
      const at::Tensor& grad_output,
      std::array<bool, 3> output_mask) override;

  virtual at::Tensor get_at_packed_weight() override;

  virtual at::Tensor pack(const at::Tensor& tensor) override;

  virtual at::Tensor to_public(const at::Tensor& tensor) override;

  virtual void may_repack(std::vector<int64_t> input_size) override;

  virtual detail::ContextConvTranspose& get_context() override;

  virtual at::Tensor get_data_handle() override;

  static c10::intrusive_ptr<ConvTransposeOpContext> create_context(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      bool weight_is_channels_last,
      std::vector<int64_t>&& input_size);
};

} // namespace cpu
} // namespace torch_ipex
