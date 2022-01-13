#include "ConvPacked.h"
#include "csrc/aten/cpu/Conv.h"
#include "csrc/aten/cpu/ParamUtils.h"
#include "csrc/aten/cpu/WeightPack.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace convolution {

c10::intrusive_ptr<ConvolutionOpContext> createConvolutionPrePackOpContext(
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
    std::vector<int64_t>&& input_size) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::createConvolutionPrePackOpContext",
      std::vector<c10::IValue>({}));
#endif
  return IpexConvolutionOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      std::move(kernel_size),
      groups,
      output_channel,
      weight_is_channels_last,
      weight_is_packed,
      std::move(input_size));
}

at::Tensor convolution_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_run", std::vector<c10::IValue>({}));
#endif
  return op_context->run(input, ideep::attr_t());
}

at::Tensor convolution_relu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_relu_run", std::vector<c10::IValue>({}));
#endif
  return op_context->run(input, ideep::attr_t::fuse_relu());
}

at::Tensor convolution_sigmoid_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_sigmoid_run", std::vector<c10::IValue>({}));
#endif
  return op_context->run(input, ideep::attr_t::fuse_sigmoid());
}

at::Tensor convolution_hardtanh_run(
    const at::Tensor& input,
    at::Scalar lower_bound,
    at::Scalar upper_bound,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_hardtanh_run", std::vector<c10::IValue>({}));
#endif
  auto lower_bound_value = lower_bound.to<float>();
  auto upper_bound_value = upper_bound.to<float>();
  return op_context->run(
      input, ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value));
}

at::Tensor convolution_elu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_elu_run", std::vector<c10::IValue>({}));
#endif
  auto alpha_value = alpha.to<float>();
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_elu(scale_value, alpha_value, input_scale_value));
}

at::Tensor convolution_swish_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_swish_run", std::vector<c10::IValue>({}));
#endif
  return op_context->run(input, ideep::attr_t::fuse_swish());
}

at::Tensor convolution_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_add_run", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::fuse_sum(scale));
}

at::Tensor convolution_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::convolution_add_relu_run", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::residual(scale));
}

ContextConvolution create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const at::IntArrayRef kernel_size,
    const int64_t groups,
    const int64_t output_channel,
    const bool weight_is_channels_last,
    const bool weight_is_packed,
    const at::IntArrayRef input_size) {
  auto dim = input_size.size() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", dim);

  bool weight_is_channels_last_ = weight_is_channels_last;
  if (!weight_is_packed) {
    weight_is_channels_last_ =
        weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
        weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  }

  auto memory_format = at::MemoryFormat::Contiguous;
  if (weight_is_channels_last_) {
    if (input_size.size() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto weight_ = weight;
  if (!weight_is_packed) {
    weight_ = weight.contiguous(memory_format);
  }

  // get original weight dims.
  std::vector<int64_t> origin_weight_dims;
  origin_weight_dims.push_back(output_channel);
  origin_weight_dims.push_back(input_size[1] / groups);
  for (auto& s : kernel_size) {
    origin_weight_dims.push_back(s);
  }
  ideep::tensor packed_weight = get_conv_packed_weight(
      weight_,
      stride_expanded,
      padding_expanded,
      dilation_expanded,
      origin_weight_dims,
      groups,
      weight_is_channels_last_,
      weight_is_packed,
      weight_is_channels_last_,
      input_size,
      ideep::attr_t());

  return ContextConvolution{
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      padding_expanded,
      stride_expanded,
      dilation_expanded,
      groups,
      weight_is_channels_last_};
}

at::Tensor run(
    const ContextConvolution& context,
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      context.weight_is_channels_last_;
  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto input_ = input.contiguous(memory_format);
  return convolution_kernel(
      input_,
      context.weight_packed_,
      context.bias_,
      context.stride_,
      context.padding_,
      context.dilation_,
      context.groups_,
      attr);
}

at::Tensor& run(
    const ContextConvolution& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      context.weight_is_channels_last_;

  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto input_ = input.contiguous(memory_format);
  // always align accumu format with inputs' format.
  accumu = accumu.contiguous(memory_format);
  convolution_kernel_output(
      input_,
      context.weight_packed_,
      context.bias_,
      accumu,
      context.stride_,
      context.padding_,
      context.dilation_,
      context.groups_,
      attr);
  return accumu;
}

} // namespace convolution
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
