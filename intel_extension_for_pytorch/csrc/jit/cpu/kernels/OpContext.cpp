#include "OpContext.h"
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
        std::vector<int64_t>&& kernel_size,
        int64_t groups,
        int64_t output_channel,
        bool weight_is_channels_last,
        std::vector<int64_t>&& input_size,
        const ideep::attr_t& attr) {
  auto op_context = torch_ipex::cpu::detail::convolution::create(
      weight,
      bias,
      stride,
      padding,
      dilation,
      kernel_size,
      groups,
      output_channel,
      weight_is_channels_last,
      input_size,
      attr);
  return c10::make_intrusive<IpexConvolutionOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      std::move(kernel_size),
      groups,
      output_channel,
      weight_is_channels_last,
      std::move(input_size),
      std::move(op_context));
}

std::vector<int64_t> ConvolutionOpContext::get_stride() {
  return stride_;
}

std::vector<int64_t> ConvolutionOpContext::get_padding() {
  return padding_;
}

std::vector<int64_t> ConvolutionOpContext::get_dilation() {
  return dilation_;
}

int64_t ConvolutionOpContext::get_groups() {
  return groups_;
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

detail::ContextConvolution& IpexConvolutionOpContext::get_conetxt() {
  return op_context_;
}

int64_t LinearOpContext::get_out_features() {
  return out_features_;
}

int64_t LinearOpContext::get_in_features() {
  return in_features_;
}

c10::optional<int64_t> LinearOpContext::get_batchsize() {
  return batch_size_;
}

c10::intrusive_ptr<LinearOpContext> IpexLinearOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    int64_t out_features,
    int64_t in_features,
    c10::optional<int64_t> batch_size) {
  auto op_context = torch_ipex::cpu::detail::linear::create(
      weight, bias, out_features, in_features, batch_size);
  return c10::make_intrusive<IpexLinearOpContext>(
      std::move(weight),
      std::move(bias),
      out_features,
      in_features,
      batch_size,
      std::move(op_context));
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
  return torch_ipex::cpu::detail::linear::get_at_packed_weight(op_context_);
}

void IpexLinearOpContext::set_bias(at::Tensor& bias) {
  return torch_ipex::cpu::detail::linear::set_bias(op_context_, bias);
}

void IpexLinearOpContext::set_weight(at::Tensor& weight) {
  return torch_ipex::cpu::detail::linear::set_weight(op_context_, weight);
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
        std::vector<int64_t>&& kernel_size,
        int64_t groups,
        int64_t output_channel,
        bool weight_is_channels_last,
        std::vector<int64_t>&& input_size) {
  auto op_context = torch_ipex::cpu::detail::conv_transpose2d::create(
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      kernel_size,
      groups,
      output_channel,
      weight_is_channels_last,
      input_size);
  return c10::make_intrusive<IpexConvTransposeOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(output_padding),
      std::move(dilation),
      std::move(kernel_size),
      std::move(input_size),
      groups,
      output_channel,
      weight_is_channels_last,
      std::move(op_context));
}

at::Tensor IpexConvTransposeOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::conv_transpose2d::run(
      op_context_, input, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexConvTransposeOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::conv_transpose2d::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexConvTransposeOpContext::get_at_packed_weight() {
  return torch_ipex::cpu::detail::conv_transpose2d::get_at_packed_weight(
      op_context_);
}

at::Tensor IpexConvTransposeOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::conv_transpose2d::pack(op_context_, tensor);
}

at::Tensor IpexConvTransposeOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::conv_transpose2d::unpack(op_context_, tensor);
}

void IpexConvTransposeOpContext::may_repack(std::vector<int64_t> input_size) {
  if (input_size_.empty() || input_size_ != input_size) {
    input_size_ = input_size;
    torch_ipex::cpu::detail::conv_transpose2d::repack_for(
        op_context_, input_size);
  }
  return;
}

} // namespace cpu
} // namespace torch_ipex
