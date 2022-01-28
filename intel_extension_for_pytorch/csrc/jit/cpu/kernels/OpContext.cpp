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
        bool weight_is_packed,
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
      weight_is_packed,
      input_size,
      attr);
  return c10::make_intrusive<IpexConvolutionOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      std::move(kernel_size),
      std::move(input_size),
      groups,
      output_channel,
      weight_is_channels_last,
      weight_is_packed,
      std::move(op_context));
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

c10::intrusive_ptr<LinearOpContext> IpexLinearOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    int64_t out_features,
    int64_t in_features,
    int64_t batch_size,
    bool weight_is_packed) {
  auto op_context = torch_ipex::cpu::detail::linear::create(
      weight, bias, out_features, in_features, batch_size, weight_is_packed);
  return c10::make_intrusive<IpexLinearOpContext>(
      std::move(weight),
      std::move(bias),
      out_features,
      in_features,
      batch_size,
      weight_is_packed,
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
        bool weight_is_packed,
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
      weight_is_packed,
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
      weight_is_packed,
      std::move(op_context));
}

at::Tensor IpexConvTransposeOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::conv_transpose2d::run(
      op_context_, input, attr);
}

} // namespace cpu
} // namespace torch_ipex
