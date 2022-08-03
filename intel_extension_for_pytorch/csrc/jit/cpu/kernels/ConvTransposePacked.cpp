#include "ConvTransposePacked.h"
#include "csrc/aten/cpu/ConvTranspose.h"
#include "csrc/aten/cpu/ParamUtils.h"
#include "csrc/aten/cpu/WeightPack.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace conv_transpose {

#define DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(FUSED_OP)             \
  at::Tensor conv_transpose_##FUSED_OP##_run(                         \
      const at::Tensor& input,                                        \
      const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) { \
    RECORD_FUNCTION(                                                  \
        "ipex_prepack::conv_transpose_" #FUSED_OP "_run",             \
        c10::ArrayRef<c10::IValue>({}));                              \
    return op_context->run(input, ideep::attr_t::fuse_##FUSED_OP());  \
  }

c10::intrusive_ptr<ConvTransposeOpContext> createConvTransposePrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    int64_t groups,
    std::vector<int64_t>&& dilation,
    bool weight_is_channels_last,
    std::vector<int64_t>&& input_size) {
  RECORD_FUNCTION(
      "ipex_prepack::createConvTransposePrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));
  return IpexConvTransposeOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(output_padding),
      std::move(dilation),
      groups,
      weight_is_channels_last,
      std::move(input_size));
}

at::Tensor conv_transpose_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_run", c10::ArrayRef<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t());
}

DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(relu);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(sigmoid);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(swish);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(tanh);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(mish);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(abs);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(exp);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(hardswish);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(square);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(log);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(round);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(sqrt);
DEFINE_CONV_TRANSPOSE_UNARY_ELTWISE_RUN(hardsigmoid);

at::Tensor conv_transpose_gelu_run(
    const at::Tensor& input,
    c10::string_view approximate,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_gelu_run", c10::ArrayRef<c10::IValue>({}));
  dnnl::algorithm gelu_type;
  if (approximate == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (approximate == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_CHECK(
        false,
        "ipex::conv_transpose_gelu_run only support tanh approximate now");
  }
  return op_context->run(
      input, ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type));
}

at::Tensor conv_transpose_leaky_relu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_leaky_relu_run",
      c10::ArrayRef<c10::IValue>({}));
  auto alpha_value = alpha.to<float>();
  return op_context->run(input, ideep::attr_t::fuse_relu(1.0, alpha_value));
}

at::Tensor conv_transpose_hardtanh_run(
    const at::Tensor& input,
    at::Scalar lower_bound,
    at::Scalar upper_bound,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_hardtanh_run",
      c10::ArrayRef<c10::IValue>({}));
  auto lower_bound_value = lower_bound.to<float>();
  auto upper_bound_value = upper_bound.to<float>();
  return op_context->run(
      input, ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value));
}

at::Tensor conv_transpose_elu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_elu_run", c10::ArrayRef<c10::IValue>({}));
  auto alpha_value = alpha.to<float>();
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_elu(scale_value, alpha_value, input_scale_value));
}

at::Tensor conv_transpose_pow_run(
    const at::Tensor& input,
    at::Scalar exponent,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_pow_run", c10::ArrayRef<c10::IValue>({}));
  auto exponent_value = exponent.to<float>();
  return op_context->run(
      input, ideep::attr_t::fuse_pow(1.0, 1.0, exponent_value));
}

at::Tensor conv_transpose_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_add_run", c10::ArrayRef<c10::IValue>({}));
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::fuse_sum(scale));
}

at::Tensor conv_transpose_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::conv_transpose_add_relu_run",
      c10::ArrayRef<c10::IValue>({}));
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::residual(scale));
}

ContextConvTranspose create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef output_padding,
    const at::IntArrayRef dilation,
    const int64_t groups,
    const bool weight_is_channels_last,
    const at::IntArrayRef input_size) {
  auto dim = weight.dim() - 2;
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto output_padding_expanded =
      expand_param_if_needed(output_padding, "output_padding", dim);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", dim);

  bool weight_is_channels_last_ = weight_is_channels_last;

  weight_is_channels_last_ =
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  auto memory_format = weight_is_channels_last_ ? weight.suggest_memory_format()
                                                : at::MemoryFormat::Contiguous;
  auto weight_ = weight.contiguous(memory_format);

  auto w = itensor_view_from_dense(weight_);
  ideep::tensor::desc ori_desc(w.get_desc());
  ideep::data_type dtype = w.get_data_type();
  // TODO: adjust padding_r
  auto expected_desc = get_conv_transpose_expected_weights_desc(
      w.get_dims(),
      dtype,
      {stride_expanded.begin(), stride_expanded.end()},
      {padding_expanded.begin(), padding_expanded.end()},
      {padding_expanded.begin(), padding_expanded.end()},
      {dilation_expanded.begin(), dilation_expanded.end()},
      groups,
      weight_is_channels_last_,
      ideep::algorithm::deconvolution_direct,
      dtype,
      input_size.vec());
  auto weight_dtype = w.get_data_type();
  expected_desc = expected_desc.to_type(weight_dtype);
  auto at_weight = empty_aten_tensor_from_desc(expected_desc, weight.options());
  ideep::tensor packed_weight;
  if (ideep::data_type::f32 == weight_dtype) {
    packed_weight.init(expected_desc, at_weight.template data_ptr<float>());
  } else {
    packed_weight.init(
        expected_desc, at_weight.template data_ptr<c10::BFloat16>());
  }

  w.transpose_(0, 1);
  packed_weight.feed_from(w, true);

  return ContextConvTranspose{
      std::move(ori_desc),
      std::move(packed_weight),
      std::move(at_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      padding_expanded,
      output_padding_expanded,
      stride_expanded,
      dilation_expanded,
      groups,
      input_size.vec(),
      weight.sizes().vec(),
      weight_is_channels_last_};
}

at::Tensor run(
    const ContextConvTranspose& context,
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
    } else if (input.dim() == 5) {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto input_ = input.contiguous(memory_format);

  return conv_transpose_kernel_impl(
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

at::Tensor& run(
    const ContextConvTranspose& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      context.weight_is_channels_last_;
  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    // TODO: support ConvTranspose1d
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else if (input.dim() == 5) {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }

  auto input_ = input.contiguous(memory_format);
  // always align accumu format with inputs' format.
  accumu = accumu.contiguous(memory_format);

  conv_transpose_out_kernel_impl(
      input_,
      context.weight_packed_,
      context.bias_,
      accumu,
      context.stride_,
      context.padding_,
      context.output_padding_,
      context.groups_,
      context.dilation_,
      context.origin_weight_dims_,
      attr);

  return accumu;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
    ContextConvTranspose& context,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask) {
  return conv_transpose_backward_kernel_impl(
      input,
      grad_output,
      context.at_weight_,
      context.weight_packed_,
      context.stride_,
      context.padding_,
      context.output_padding_,
      context.groups_,
      context.dilation_,
      output_mask,
      context.weight_is_channels_last_);
}

at::Tensor get_at_packed_weight(ContextConvTranspose& context) {
  return context.at_weight_;
}

at::Tensor pack(ContextConvTranspose& context, const at::Tensor& tensor) {
  auto ideep_tensor = itensor_view_from_dense(tensor);
  auto dtype = ideep_tensor.get_data_type();
  auto expected_desc = context.weight_packed_.get_desc().to_type(dtype);
  auto packed_at_tensor =
      empty_aten_tensor_from_desc(expected_desc, tensor.options());
  ideep::tensor packed_tensor;
  if (ideep::data_type::f32 == dtype) {
    packed_tensor.init(
        expected_desc, packed_at_tensor.template data_ptr<float>());
  } else {
    packed_tensor.init(
        expected_desc, packed_at_tensor.template data_ptr<c10::BFloat16>());
  }
  ideep_tensor.transpose_(0, 1);
  packed_tensor.feed_from(ideep_tensor, true);
  return packed_at_tensor;
}

at::Tensor unpack(ContextConvTranspose& context, const at::Tensor& tensor) {
  auto dtype = get_mkldnn_dtype(tensor.scalar_type());
  auto expected_desc = context.weight_packed_.get_desc().to_type(dtype);
  ideep::tensor blocked_tensor;
  if (ideep::data_type::f32 == dtype) {
    blocked_tensor.init(expected_desc, tensor.template data_ptr<float>());
  } else {
    blocked_tensor.init(
        expected_desc, tensor.template data_ptr<c10::BFloat16>());
  }

  at::Tensor result = at::empty(context.origin_weight_dims_, tensor.options());
  if (context.weight_is_channels_last_) {
    if (context.original_desc_.get_ndims() == 4) {
      result = result.to(at::MemoryFormat::ChannelsLast);
    } else if (context.original_desc_.get_ndims() == 5) {
      result = result.to(at::MemoryFormat::ChannelsLast3d);
    }
  }
  ideep::tensor pub_tensor = itensor_view_from_dense(result);
  auto pub_tensor_desc = context.original_desc_.to_type(dtype);
  if (ideep::data_type::f32 == dtype) {
    pub_tensor.init(pub_tensor_desc, result.template data_ptr<float>());
  } else {
    pub_tensor.init(pub_tensor_desc, result.template data_ptr<c10::BFloat16>());
  }
  pub_tensor.transpose_(0, 1);
  pub_tensor.feed_from(blocked_tensor, true);
  return result;
}

void repack_for(
    ContextConvTranspose& context,
    std::vector<int64_t> input_size) {
  auto dtype = context.original_desc_.get_data_type();
  ideep::tensor packed_weight;
  auto packed_desc = get_conv_transpose_expected_weights_desc(
      context.origin_weight_dims_,
      dtype,
      context.stride_,
      context.padding_,
      context.padding_,
      context.dilation_,
      context.groups_,
      context.weight_is_channels_last_,
      ideep::algorithm::deconvolution_direct,
      dtype,
      input_size);
  auto new_at_weight =
      empty_aten_tensor_from_desc(packed_desc, context.at_weight_.options());
  if (ideep::data_type::f32 == dtype) {
    packed_weight.init(packed_desc, new_at_weight.template data_ptr<float>());
  } else {
    packed_weight.init(
        packed_desc, new_at_weight.template data_ptr<c10::BFloat16>());
  }
  packed_weight.feed_from(context.weight_packed_);
  context.at_weight_ = new_at_weight;
  context.weight_packed_ = packed_weight;
}

} // namespace conv_transpose
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
