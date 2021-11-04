#include "ConvTransposePacked.h"
#include "cpu/ConvTranspose.h"
#include "cpu/ParamUtils.h"
#include "cpu/WeightPack.h"
#include "cpu/mkldnn/MKLDNNCommon.h"
#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace conv_transpose2d {

c10::intrusive_ptr<ConvTransposeOpContext> createConvTransposePrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    int64_t groups,
    std::vector<int64_t>&& dilation,
    std::vector<int64_t>&& kernel_size,
    int64_t output_channel,
    bool weight_is_channels_last,
    bool weight_is_packed,
    std::vector<int64_t>&& input_size) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::createConvTransposePrePackOpContext",
      std::vector<c10::IValue>({}));
#endif
  return IpexConvTransposeOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(output_padding),
      std::move(dilation),
      std::move(kernel_size),
      groups,
      output_channel,
      weight_is_channels_last,
      weight_is_packed,
      std::move(input_size));
}

at::Tensor conv_transpose2d_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose2d_run", std::vector<c10::IValue>({}));
#endif
  return op_context->run(input, ideep::attr_t());
}

ContextConvTranspose create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef output_padding,
    const at::IntArrayRef dilation,
    const at::IntArrayRef kernel_size,
    const int64_t groups,
    const int64_t output_channel,
    const bool weight_is_channels_last,
    const bool weight_is_packed,
    const at::IntArrayRef input_size) {
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  const auto output_padding_expanded =
      expand_param_if_needed(output_padding, "output_padding", 2);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", 2);

  bool weight_is_channels_last_ = weight_is_channels_last;

  if (!weight_is_packed) {
    weight_is_channels_last_ =
        weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  }
  auto memory_format = weight_is_channels_last_ ? at::MemoryFormat::ChannelsLast
                                                : at::MemoryFormat::Contiguous;
  auto weight_ = weight;
  if (!weight_is_packed) {
    weight_ = weight.contiguous(memory_format);
  }

  // get original weight dims.
  std::vector<int64_t> origin_weight_dims;
  if (weight_is_packed) {
    origin_weight_dims.push_back(input_size[1]);
    origin_weight_dims.push_back(output_channel / groups);
    for (auto& s : kernel_size) {
      origin_weight_dims.push_back(s);
    }
  } else {
    for (auto& s : weight.sizes()) {
      origin_weight_dims.push_back(s);
    }
  }

  ideep::tensor packed_weight = get_conv_transpose2d_packed_weight(
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

  return ContextConvTranspose{
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      {output_channel, input_size[1], kernel_size[0], kernel_size[1]},
      {padding_expanded[0], padding_expanded[1]},
      {output_padding_expanded[0], output_padding_expanded[1]},
      {stride_expanded[0], stride_expanded[1]},
      {dilation_expanded[0], dilation_expanded[1]},
      groups,
      {input_size[0], input_size[1], input_size[2], input_size[3]},
      {origin_weight_dims[0],
       origin_weight_dims[1],
       origin_weight_dims[2],
       origin_weight_dims[3]},
      weight_is_channels_last_};
}

at::Tensor run(
    const ContextConvTranspose& context,
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      context.weight_is_channels_last_;
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast
                                         : at::MemoryFormat::Contiguous;
  auto input_ = input.contiguous(memory_format);

  return conv_transpose2d_kernel_impl(
      input_,
      context.weight_packed_,
      context.bias_,
      context.stride_,
      context.padding_,
      context.output_padding_,
      context.groups_,
      context.dilation_,
      context.origin_weight_dims_,
      attr);
}

} // namespace conv_transpose2d
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
