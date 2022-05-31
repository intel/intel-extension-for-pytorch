#include "OpContext.h"
#include <torch/extension.h>
#include "ConvPacked.h"
#include "ConvTransposePacked.h"
#include "LinearPacked.h"

namespace torch_ipex {
namespace cpu {

c10::intrusive_ptr<ConvolutionOpContext> IpexConvolutionOpContext::
    create_context(
        at::Tensor&& weight,
        c10::optional<at::Tensor>&& bias,
        std::vector<int64_t>&& stride,
        std::vector<int64_t>&& padding,
        std::vector<int64_t>&& dilation,
        int64_t groups,
        bool weight_is_channels_last,
        std::vector<int64_t>&& input_size,
        const ideep::attr_t& attr) {
  auto op_context = torch_ipex::cpu::detail::convolution::create(
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      weight_is_channels_last,
      input_size,
      attr);
  return c10::make_intrusive<IpexConvolutionOpContext>(
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      std::move(input_size),
      std::move(op_context));
}

std::vector<int64_t> ConvolutionOpContext::get_stride() {
  return this->get_context().stride_;
}

std::vector<int64_t> ConvolutionOpContext::get_padding() {
  return this->get_context().padding_;
}

std::vector<int64_t> ConvolutionOpContext::get_dilation() {
  return this->get_context().dilation_;
}

int64_t ConvolutionOpContext::get_groups() {
  return this->get_context().groups_;
}

at::Tensor IpexConvolutionOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::convolution::run(op_context_, input, attr);
}

at::Tensor& IpexConvolutionOpContext::run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::convolution::run(
      op_context_, input, accumu, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexConvolutionOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::convolution::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexConvolutionOpContext::get_at_packed_weight() {
  return torch_ipex::cpu::detail::convolution::get_at_packed_weight(
      op_context_);
}

at::Tensor IpexConvolutionOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::convolution::pack(op_context_, tensor);
}

at::Tensor IpexConvolutionOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::convolution::unpack(op_context_, tensor);
}

detail::ContextConvolution& IpexConvolutionOpContext::get_context() {
  return op_context_;
}

at::Tensor IpexConvolutionOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr.data_ptr<int64_t>()[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

c10::optional<int64_t> LinearOpContext::get_batchsize() {
  return batch_size_;
}

c10::intrusive_ptr<LinearOpContext> IpexLinearOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size) {
  auto op_context =
      torch_ipex::cpu::detail::linear::create(weight, bias, batch_size);
  return c10::make_intrusive<IpexLinearOpContext>(
      batch_size,
      std::move(op_context));
}

at::Tensor IpexLinearOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr.data_ptr<int64_t>()[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

at::Tensor IpexLinearOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::linear::run(op_context_, input, attr);
}

at::Tensor& IpexLinearOpContext::run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::linear::run(op_context_, input, accumu, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexLinearOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::linear::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexLinearOpContext::get_at_packed_weight() {
  return op_context_.at_weight_;
}

int64_t IpexLinearOpContext::get_out_features() {
  return op_context_.original_desc_.get_dim(0);
}

int64_t IpexLinearOpContext::get_in_features() {
  return op_context_.original_desc_.get_dim(1);
}

detail::ContextLinear& IpexLinearOpContext::get_context() {
  return op_context_;
}

void IpexLinearOpContext::set_bias(at::Tensor& bias) {
  op_context_.bias_ = c10::make_optional<at::Tensor>(std::move(bias));
}

void IpexLinearOpContext::set_weight(at::Tensor& weight) {
  op_context_.at_weight_.copy_(weight);
}

at::Tensor IpexLinearOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::linear::pack(op_context_, tensor);
}

at::Tensor IpexLinearOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::linear::unpack(op_context_, tensor);
}

void IpexLinearOpContext::may_repack(int64_t batch_size) {
  if (!batch_size_.has_value() || batch_size_.has_value() != batch_size) {
    batch_size_ = c10::make_optional(batch_size);
    torch_ipex::cpu::detail::linear::repack_for(op_context_, batch_size);
  }
  return;
}

c10::intrusive_ptr<ConvTransposeOpContext> IpexConvTransposeOpContext::
    create_context(
        at::Tensor&& weight,
        c10::optional<at::Tensor>&& bias,
        std::vector<int64_t>&& stride,
        std::vector<int64_t>&& padding,
        std::vector<int64_t>&& output_padding,
        std::vector<int64_t>&& dilation,
        int64_t groups,
        bool weight_is_channels_last,
        std::vector<int64_t>&& input_size) {
  auto op_context = torch_ipex::cpu::detail::conv_transpose::create(
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      weight_is_channels_last,
      input_size);
  return c10::make_intrusive<IpexConvTransposeOpContext>(
      std::move(stride),
      std::move(padding),
      std::move(output_padding),
      std::move(dilation),
      std::move(input_size),
      std::move(op_context));
}

at::Tensor IpexConvTransposeOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::conv_transpose::run(op_context_, input, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexConvTransposeOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::conv_transpose::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexConvTransposeOpContext::get_at_packed_weight() {
  return torch_ipex::cpu::detail::conv_transpose::get_at_packed_weight(
      op_context_);
}

at::Tensor IpexConvTransposeOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::conv_transpose::pack(op_context_, tensor);
}

at::Tensor IpexConvTransposeOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::conv_transpose::unpack(op_context_, tensor);
}

void IpexConvTransposeOpContext::may_repack(std::vector<int64_t> input_size) {
  if (input_size_.empty() || input_size_ != input_size) {
    input_size_ = input_size;
    torch_ipex::cpu::detail::conv_transpose::repack_for(
        op_context_, input_size);
  }
  return;
}

at::Tensor IpexConvTransposeOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr.data_ptr<int64_t>()[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

detail::ContextConvTranspose& IpexConvTransposeOpContext::get_context() {
  return op_context_;
}

} // namespace cpu
} // namespace torch_ipex
