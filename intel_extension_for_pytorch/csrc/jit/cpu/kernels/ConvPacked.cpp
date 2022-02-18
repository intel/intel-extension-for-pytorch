#include "ConvPacked.h"
#include <dnnl.hpp>
#include "csrc/aten/cpu/Conv.h"
#include "csrc/aten/cpu/ParamUtils.h"
#include "csrc/aten/cpu/WeightPack.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/cpu/ideep/ideep/utils.hpp"
#include "csrc/utils/ipex_op_profile.h"

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
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::createConvolutionPrePackOpContext",
      std::vector<c10::IValue>({}));
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
      std::move(input_size),
      ideep::attr_t());
}

at::Tensor convolution_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_run", std::vector<c10::IValue>({}));
  return op_context->run(input, ideep::attr_t());
}

at::Tensor convolution_relu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_relu_run", std::vector<c10::IValue>({}));
  return op_context->run(input, ideep::attr_t::fuse_relu());
}

at::Tensor convolution_sigmoid_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_sigmoid_run", std::vector<c10::IValue>({}));
  return op_context->run(input, ideep::attr_t::fuse_sigmoid());
}

at::Tensor convolution_hardtanh_run(
    const at::Tensor& input,
    at::Scalar lower_bound,
    at::Scalar upper_bound,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_hardtanh_run", std::vector<c10::IValue>({}));
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
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_elu_run", std::vector<c10::IValue>({}));
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
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_swish_run", std::vector<c10::IValue>({}));
  return op_context->run(input, ideep::attr_t::fuse_swish());
}

at::Tensor convolution_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_add_run", std::vector<c10::IValue>({}));
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::fuse_sum(scale));
}

at::Tensor convolution_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_add_relu_run", std::vector<c10::IValue>({}));
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::residual(scale));
}

at::Tensor& convolution_bottleneck_run(
    at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context1,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context2,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context3) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_bottleneck_runi_v1",
      std::vector<c10::IValue>({}));

  auto memory_format = input.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                        : at::MemoryFormat::ChannelsLast3d;
  input = input.contiguous(memory_format);

  auto& context1 = op_context1->get_conetxt();
  auto& context2 = op_context2->get_conetxt();
  auto& context3 = op_context3->get_conetxt();
  if (input.sizes().vec() == context1.conv_params_.pd.src_desc().dims() &&
      omp_get_max_threads() == context1.conv_params_.pd_use_threads) {
    auto mkldnn_input = dnnl::memory(
        context1.conv_params_.pd.src_desc(),
        ideep::engine::cpu_engine(),
        input.data_ptr());
    auto ouput1 = dnnl::memory(
        context1.conv_params_.pd.dst_desc(), ideep::engine::cpu_engine());
    auto ouput2 = dnnl::memory(
        context2.conv_params_.pd.dst_desc(), ideep::engine::cpu_engine());

    auto desc = context1.conv_params_.pd.scratchpad_desc();
    if (context2.conv_params_.pd.scratchpad_desc().get_size() >
        desc.get_size()) {
      desc = context2.conv_params_.pd.scratchpad_desc();
    }
    if (context3.conv_params_.pd.scratchpad_desc().get_size() >
        desc.get_size()) {
      desc = context3.conv_params_.pd.scratchpad_desc();
    }

    auto scratchpad = dnnl::memory(desc, ideep::engine::cpu_engine());

    context1.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, mkldnn_input},
         {DNNL_ARG_WEIGHTS, context1.weight_packed_},
         {DNNL_ARG_BIAS, context1.bias_},
         {DNNL_ARG_DST, ouput1},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    context2.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, ouput1},
         {DNNL_ARG_WEIGHTS, context2.weight_packed_},
         {DNNL_ARG_BIAS, context2.bias_},
         {DNNL_ARG_DST, ouput2},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    context3.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, ouput2},
         {DNNL_ARG_WEIGHTS, context3.weight_packed_},
         {DNNL_ARG_BIAS, context3.bias_},
         {DNNL_ARG_DST, mkldnn_input},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    return input;
  } else {
    auto output1 = run(context1, input, context1.conv_params_.op_attr);
    auto output2 = run(context2, output1, context2.conv_params_.op_attr);
    return run(context3, output2, input, context3.conv_params_.op_attr);
  }
}

at::Tensor convolution_bottleneck_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context1,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context2,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context3,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context4) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::convolution_bottleneck_run_v2",
      std::vector<c10::IValue>({}));

  auto memory_format = input.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                        : at::MemoryFormat::ChannelsLast3d;
  auto input_ = input.contiguous(memory_format);

  auto& context1 = op_context1->get_conetxt();
  auto& context2 = op_context2->get_conetxt();
  auto& context4 = op_context4->get_conetxt();
  auto& context3 = op_context3->get_conetxt();

  if (input_.sizes().vec() == context1.conv_params_.pd.src_desc().dims() &&
      omp_get_max_threads() == context1.conv_params_.pd_use_threads) {
    auto mkldnn_input = dnnl::memory(
        context1.conv_params_.pd.src_desc(),
        ideep::engine::cpu_engine(),
        input.data_ptr());

    auto ouput1 = dnnl::memory(
        context1.conv_params_.pd.dst_desc(), ideep::engine::cpu_engine());
    auto ouput2 = dnnl::memory(
        context2.conv_params_.pd.dst_desc(), ideep::engine::cpu_engine());

    auto result = at::empty(
        context3.conv_params_.pd.dst_desc().dims(),
        input_.options().memory_format(input_.suggest_memory_format()));

    auto ouput3 = dnnl::memory(
        context3.conv_params_.pd.dst_desc(),
        ideep::engine::cpu_engine(),
        result.data_ptr());

    auto desc = context1.conv_params_.pd.scratchpad_desc();
    if (context2.conv_params_.pd.scratchpad_desc().get_size() >
        desc.get_size()) {
      desc = context2.conv_params_.pd.scratchpad_desc();
    }
    if (context3.conv_params_.pd.scratchpad_desc().get_size() >
        desc.get_size()) {
      desc = context3.conv_params_.pd.scratchpad_desc();
    }
    if (context4.conv_params_.pd.scratchpad_desc().get_size() >
        desc.get_size()) {
      desc = context4.conv_params_.pd.scratchpad_desc();
    }
    auto scratchpad = dnnl::memory(desc, ideep::engine::cpu_engine());
    context1.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, mkldnn_input},
         {DNNL_ARG_WEIGHTS, context1.weight_packed_},
         {DNNL_ARG_BIAS, context1.bias_},
         {DNNL_ARG_DST, ouput1},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    context2.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, ouput1},
         {DNNL_ARG_WEIGHTS, context2.weight_packed_},
         {DNNL_ARG_BIAS, context2.bias_},
         {DNNL_ARG_DST, ouput2},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    context3.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, mkldnn_input},
         {DNNL_ARG_WEIGHTS, context3.weight_packed_},
         {DNNL_ARG_BIAS, context3.bias_},
         {DNNL_ARG_DST, ouput3},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    context4.conv_desc_.execute(
        ideep::stream::default_stream(),
        {{DNNL_ARG_SRC, ouput2},
         {DNNL_ARG_WEIGHTS, context4.weight_packed_},
         {DNNL_ARG_BIAS, context4.bias_},
         {DNNL_ARG_DST, ouput3},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    return result;
  } else {
    auto output1 = run(context1, input, context1.conv_params_.op_attr);
    auto output2 = run(context2, output1, context2.conv_params_.op_attr);
    auto output3 = run(context3, input, context3.conv_params_.op_attr);
    return run(context4, output2, output3, context4.conv_params_.op_attr);
  }
}

ContextConvolution create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const at::IntArrayRef kernel_size,
    const int64_t groups,
    const int64_t output_channel,
    const bool weight_is_channels_last,
    const bool weight_is_packed,
    const at::IntArrayRef input_size,
    const ideep::attr_t& attr) {
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
  auto format_tag = input_size.size() == 4 ? ideep::format_tag::nchw
                                           : ideep::format_tag::ncdhw;
  if (weight_is_channels_last_) {
    if (input_size.size() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
      format_tag = ideep::format_tag::nhwc;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
      format_tag = ideep::format_tag::ndhwc;
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

  ideep::convolution_forward_params conv_params;
  std::vector<int64_t> output_sizes = calc_conv_output_size(
      input_size,
      origin_weight_dims,
      padding_expanded,
      stride_expanded,
      dilation_expanded);

  // src and weight always have same dtype and data format.
  auto data_type = get_mkldnn_dtype(weight_.scalar_type());

  ideep::tensor src = ideep::tensor(
      {input_size.begin(), input_size.end()}, data_type, format_tag);
  ideep::tensor dst = ideep::tensor(
      {output_sizes.begin(), output_sizes.end()}, data_type, format_tag);

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  ideep::tensor mkldnn_bias;
  if (bias.defined()) {
    mkldnn_bias = itensor_view_from_dense(bias);
    ideep::convolution_forward::prepare(
        conv_params,
        src,
        packed_weight,
        mkldnn_bias,
        {output_sizes.begin(), output_sizes.end()},
        dst,
        {stride_expanded.begin(), stride_expanded.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference);
  } else {
    ideep::convolution_forward::prepare(
        conv_params,
        src,
        packed_weight,
        {output_sizes.begin(), output_sizes.end()},
        dst,
        {stride_expanded.begin(), stride_expanded.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference);
  }
  return ContextConvolution{
      std::move(packed_weight),
      std::move(mkldnn_bias),
      padding_expanded,
      stride_expanded,
      dilation_expanded,
      groups,
      weight_is_channels_last_,
      conv_params,
      ideep::convolution_forward::super(conv_params.pd)};
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
  if (input_.sizes().vec() == context.conv_params_.pd.src_desc().dims() &&
      attr == context.conv_params_.op_attr &&
      omp_get_max_threads() == context.conv_params_.pd_use_threads) {
    auto output = at::empty(
        context.conv_params_.pd.dst_desc().dims(),
        input_.options().memory_format(input_.suggest_memory_format()));
    const ideep::tensor mkldnn_input = itensor_view_from_dense(input_);
    ideep::tensor mkldnn_output = itensor_view_from_dense(output);
    if (context.bias_.is_empty()) {
      ideep::convolution_forward::compute(
          context.conv_params_,
          context.conv_desc_,
          mkldnn_input,
          context.weight_packed_,
          mkldnn_output);
    } else {
      ideep::convolution_forward::compute(
          context.conv_params_,
          context.conv_desc_,
          mkldnn_input,
          context.weight_packed_,
          context.bias_,
          mkldnn_output);
    }
    return output;
  }
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
  if (input_.sizes().vec() == context.conv_params_.pd.src_desc().dims() &&
      attr == context.conv_params_.op_attr &&
      omp_get_max_threads() == context.conv_params_.pd_use_threads) {
    const ideep::tensor mkldnn_input = itensor_view_from_dense(input_);
    ideep::tensor mkldnn_output = itensor_view_from_dense(accumu);

    if (context.bias_.is_empty()) {
      ideep::convolution_forward::compute(
          context.conv_params_,
          context.conv_desc_,
          mkldnn_input,
          context.weight_packed_,
          mkldnn_output);
    } else {
      ideep::convolution_forward::compute(
          context.conv_params_,
          context.conv_desc_,
          mkldnn_input,
          context.weight_packed_,
          context.bias_,
          mkldnn_output);
    }
  } else {
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
  }
  return accumu;
}

} // namespace convolution
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
