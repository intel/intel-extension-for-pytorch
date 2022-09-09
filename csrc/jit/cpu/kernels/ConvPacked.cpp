#include "ConvPacked.h"
#include <dnnl.hpp>
#include "aten/Conv.h"
#include "aten/ParamUtils.h"
#include "aten/WeightPack.h"
#include "aten/utils/utils.h"
#include "ideep/IDeepConversions.h"
#include "ideep/ideep.hpp"
#include "ideep/ideep/utils.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace convolution {

#define DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(FUSED_OP)               \
  at::Tensor convolution_##FUSED_OP##_run(                           \
      const at::Tensor& input,                                       \
      const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {  \
    RECORD_FUNCTION(                                                 \
        "ipex_prepack::convolution_" #FUSED_OP "_run",               \
        c10::ArrayRef<c10::IValue>({}));                             \
    return op_context->run(input, ideep::attr_t::fuse_##FUSED_OP()); \
  }

// follow check rules from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp
static void check_shape_forward(
    const at::IntArrayRef& input_sizes,
    const at::IntArrayRef& weight_sizes,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& dilation,
    const int64_t groups) {
#define MKLDNN_CONV_ARG_CHECK(IT, OP) \
  std::any_of(IT.begin(), IT.end(), [](auto x) { return x OP 0; })
  auto is_padding_neg = MKLDNN_CONV_ARG_CHECK(padding, <);
  auto is_stride_nonpos = MKLDNN_CONV_ARG_CHECK(stride, <=);
  auto is_dilation_nonpos = MKLDNN_CONV_ARG_CHECK(dilation, <=);
#undef MKLDNN_CONV_ARG_CHECK
  TORCH_CHECK(!is_padding_neg, "negative padding is not supported");
  TORCH_CHECK(!is_stride_nonpos, "non-positive stride is not supported");
  TORCH_CHECK(!is_dilation_nonpos, "non-positive dilation is not supported");
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");

  int64_t k = input_sizes.size();
  int64_t weight_dim = weight_sizes.size();

  TORCH_CHECK(
      weight_dim == k,
      "Expected ",
      weight_dim,
      "-dimensional input for ",
      weight_dim,
      "-dimensional weight ",
      weight_sizes,
      ", but got ",
      k,
      "-dimensional input of size ",
      input_sizes,
      " instead");
  TORCH_CHECK(
      weight_sizes[0] >= groups,
      "Given groups=",
      groups,
      ", expected weight to be at least ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");
  TORCH_CHECK(
      weight_sizes[0] % groups == 0,
      "Given groups=",
      groups,
      ", expected weight to be divisible by ",
      groups,
      " at dimension 0, but got weight of size [",
      weight_sizes,
      "] instead");
  TORCH_CHECK(
      input_sizes[1] == (weight_sizes[1] * groups),
      "Given groups=",
      groups,
      ", weight of size ",
      weight_sizes,
      ", expected input",
      input_sizes,
      " to have ",
      (weight_sizes[1] * groups),
      " channels, but got ",
      input_sizes[1],
      " channels instead");
  TORCH_CHECK(
      !bias.has_value() ||
          (bias.value().ndimension() == 1 &&
           bias.value().size(0) == weight_sizes[0]),
      "Given weight of size ",
      weight_sizes,
      ", expected bias to be 1-dimensional with ",
      weight_sizes[0],
      " elements",
      ", but got bias of size ",
      bias.value().sizes(),
      " instead");

  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel_shape;
  bool kernel_size_correct = true;

  for (const auto i : c10::irange(2, k)) {
    input_shape.push_back(input_sizes[i] + 2 * padding[i - 2]);
    // log new kernel size considering dilation
    kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
    if (input_shape.back() < kernel_shape.back()) {
      kernel_size_correct = false;
    }
  }

  TORCH_CHECK(
      input_shape.size() == kernel_shape.size(),
      "Inconsistent shape between Input and Kernel");

  if (!kernel_size_correct) {
    // If kernel size is incorrect
    std::ostringstream input_ss;
    std::ostringstream kernel_ss;
    std::string separator = "";

    for (int i = 0, len = input_shape.size(); i < len; ++i) {
      input_ss << separator << input_shape[i];
      kernel_ss << separator << kernel_shape[i];
      separator = " x ";
    }

    TORCH_CHECK(
        false,
        "Calculated padded input size per channel: (",
        input_ss.str(),
        "). "
        "Kernel size: (",
        kernel_ss.str(),
        "). Kernel size can't be greater than actual input size");
  }
}

c10::intrusive_ptr<ConvolutionOpContext> createConvolutionPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    bool weight_is_channels_last,
    std::vector<int64_t>&& input_size) {
  RECORD_FUNCTION(
      "ipex_prepack::createConvolutionPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));
  return IpexConvolutionOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      groups,
      weight_is_channels_last,
      std::move(input_size),
      ideep::attr_t());
}

at::Tensor convolution_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_run", c10::ArrayRef<c10::IValue>({}));
  return op_context->run(input, ideep::attr_t());
}

DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(relu);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(sigmoid);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(swish);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(tanh);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(mish);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(abs);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(exp);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(hardswish);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(square);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(log);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(round);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(sqrt);
DEFINE_CONVOLUTION_UNARY_ELTWISE_RUN(hardsigmoid);

at::Tensor convolution_leaky_relu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_leaky_relu_run",
      c10::ArrayRef<c10::IValue>({}));
  auto alpha_value = alpha.to<float>();
  return op_context->run(input, ideep::attr_t::fuse_relu(1.0, alpha_value));
}

at::Tensor convolution_hardtanh_run(
    const at::Tensor& input,
    at::Scalar lower_bound,
    at::Scalar upper_bound,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_hardtanh_run", c10::ArrayRef<c10::IValue>({}));
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
  RECORD_FUNCTION(
      "ipex_prepack::convolution_elu_run", c10::ArrayRef<c10::IValue>({}));
  auto alpha_value = alpha.to<float>();
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_elu(scale_value, alpha_value, input_scale_value));
}

at::Tensor convolution_pow_run(
    const at::Tensor& input,
    at::Scalar exponent,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_pow_run", c10::ArrayRef<c10::IValue>({}));
  auto exponent_value = exponent.to<float>();
  return op_context->run(
      input, ideep::attr_t::fuse_pow(1.0, 1.0, exponent_value));
}

at::Tensor convolution_gelu_run(
    const at::Tensor& input,
    const c10::string_view approximate,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_gelu_run", c10::ArrayRef<c10::IValue>({}));
  // https://github.com/pytorch/pytorch/pull/61439
  // at::gelu can support tanh approximate now and OneDNN also support it
  // by changing algorithm If there is other type of approximate are added to
  // pytorch while  OneDNN not support it, we might need a fallback path here.
  dnnl::algorithm gelu_type;
  if (approximate == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (approximate == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_CHECK(
        false, "ipex::linear_gelu_run only support tanh approximate now");
  }
  return op_context->run(
      input, ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type));
}

at::Tensor convolution_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_add_run", c10::ArrayRef<c10::IValue>({}));
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::fuse_sum(scale));
}

at::Tensor convolution_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_add_relu_run", c10::ArrayRef<c10::IValue>({}));
  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::residual(scale));
}

at::Tensor& convolution_bottleneck_run(
    at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context1,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context2,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context3) {
  RECORD_FUNCTION(
      "ipex_prepack::convolution_bottleneck_runi_v1",
      c10::ArrayRef<c10::IValue>({}));

  auto memory_format = input.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                        : at::MemoryFormat::ChannelsLast3d;
  input = input.contiguous(memory_format);

  auto& context1 = op_context1->get_context();
  auto& context2 = op_context2->get_context();
  auto& context3 = op_context3->get_context();
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
  RECORD_FUNCTION(
      "ipex_prepack::convolution_bottleneck_run_v2",
      c10::ArrayRef<c10::IValue>({}));

  auto memory_format = input.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                        : at::MemoryFormat::ChannelsLast3d;
  auto input_ = input.contiguous(memory_format);

  auto& context1 = op_context1->get_context();
  auto& context2 = op_context2->get_context();
  auto& context4 = op_context4->get_context();
  auto& context3 = op_context3->get_context();

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
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const int64_t groups,
    const bool weight_is_channels_last,
    const std::vector<int64_t>& input_size_,
    const ideep::attr_t& attr) {
  auto input_size = input_size_.empty()
      ? gen_dummy_input_size_for(weight.sizes(), groups)
      : input_size_;
  auto dim = input_size.size() - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", dim);

  check_shape_forward(
      input_size,
      weight.sizes(),
      bias,
      padding_expanded,
      stride_expanded,
      dilation_expanded,
      groups);

  bool weight_is_channels_last_ = weight_is_channels_last;
  weight_is_channels_last_ =
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;

  auto memory_format = at::MemoryFormat::Contiguous;
  auto format_tag = ideep::format_tag::nchw;
  if (input_size.size() == 5) {
    format_tag = ideep::format_tag::ncdhw;
  } else if (input_size.size() == 3) {
    format_tag = ideep::format_tag::nwc;
  }
  if (weight_is_channels_last_) {
    if (input_size.size() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
      format_tag = ideep::format_tag::nhwc;
    } else if (input_size.size() == 5) {
      memory_format = at::MemoryFormat::ChannelsLast3d;
      format_tag = ideep::format_tag::ndhwc;
    }
  }
  auto weight_ = weight;
  weight_ = weight.contiguous(memory_format);
  auto w = itensor_view_from_dense(weight_);
  ideep::convolution_forward_params conv_params;
  std::vector<int64_t> output_sizes = calc_conv_output_size(
      input_size,
      weight.sizes().vec(),
      padding_expanded,
      stride_expanded,
      dilation_expanded);

  // src and weight always have same dtype and data format.
  auto data_type = get_mkldnn_dtype(weight_.scalar_type());

  ideep::tensor src = ideep::tensor(
      {input_size.begin(), input_size.end()}, data_type, format_tag);
  ideep::tensor dst = ideep::tensor(
      {output_sizes.begin(), output_sizes.end()}, data_type, format_tag);

  ideep::tensor mkldnn_bias;
  if (bias.has_value() && bias.value().defined()) {
    mkldnn_bias = itensor_view_from_dense(bias.value());
    ideep::convolution_forward::prepare(
        conv_params,
        src,
        w,
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
        w,
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
  ideep::tensor::desc ori_desc(w.get_desc());
  ideep::data_type dtype = w.get_data_type();
  auto expected_desc =
      ideep::tensor::desc(conv_params.pd.weights_desc(), groups);
  auto at_weight = empty_aten_tensor_from_desc(expected_desc, weight.options());
  ideep::tensor packed_weight;
  if (ideep::data_type::f32 == dtype) {
    packed_weight.init(expected_desc, at_weight.template data_ptr<float>());
  } else {
    packed_weight.init(
        expected_desc, at_weight.template data_ptr<c10::BFloat16>());
  }
  packed_weight.feed_from(w);

  return ContextConvolution{
      std::move(ori_desc),
      std::move(packed_weight),
      std::move(mkldnn_bias),
      std::move(at_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
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
    } else if (input.dim() == 5) {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto input_ = input;
  if (!is_channels_last_1d(input)) {
    input_ = input.contiguous(memory_format);
  }

  check_shape_forward(
      input_.sizes(),
      context.weight_packed_.get_dims(),
      context.at_bias_,
      context.padding_,
      context.stride_,
      context.dilation_,
      context.groups_);

  if (input_.sizes().vec() == context.conv_params_.pd.src_desc().dims() &&
      attr == context.conv_params_.op_attr &&
      omp_get_max_threads() == context.conv_params_.pd_use_threads) {
    auto output_sizes = context.conv_params_.pd.dst_desc().dims();
    auto output = at::empty(
        output_sizes,
        input_.options().memory_format(input_.suggest_memory_format()));
    if (input.dim() == 3) {
      std::vector<int64_t> output_strides = {
          (output_sizes[1] * output_sizes[2]), 1, output_sizes[1]};
      output =
          at::empty_strided(output_sizes, output_strides, input_.options());
    }

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
    } else if (input.dim() == 5) {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto input_ = input;
  if (!is_channels_last_1d(input)) {
    input_ = input.contiguous(memory_format);
    if (input.dim() == 3) {
      input_ = to_channels_last_1d(input_);
    }
  }

  // always align accumu format with inputs' format.
  if (!is_channels_last_1d(accumu)) {
    accumu = accumu.contiguous(memory_format);
    if (input.dim() == 3) {
      accumu = to_channels_last_1d(accumu);
    }
  }

  check_shape_forward(
      input_.sizes(),
      context.weight_packed_.get_dims(),
      context.at_bias_,
      context.padding_,
      context.stride_,
      context.dilation_,
      context.groups_);

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
    ContextConvolution& context,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask) {
  return convolution_backward_kernel(
      input,
      grad_output,
      context.at_weight_,
      context.weight_packed_,
      context.bias_,
      context.stride_,
      context.padding_,
      context.dilation_,
      context.groups_,
      context.weight_is_channels_last_,
      output_mask);
}

at::Tensor get_at_packed_weight(ContextConvolution& context) {
  return context.at_weight_;
}

at::Tensor pack(ContextConvolution& context, const at::Tensor& tensor) {
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
  packed_tensor.feed_from(ideep_tensor);
  return packed_at_tensor;
}

at::Tensor unpack(ContextConvolution& context, const at::Tensor& tensor) {
  auto dtype = get_mkldnn_dtype(tensor.scalar_type());
  auto expected_desc = context.weight_packed_.get_desc().to_type(dtype);
  ideep::tensor blocked_tensor;
  if (ideep::data_type::f32 == dtype) {
    blocked_tensor.init(expected_desc, tensor.template data_ptr<float>());
  } else {
    blocked_tensor.init(
        expected_desc, tensor.template data_ptr<c10::BFloat16>());
  }

  at::Tensor result = at::empty(expected_desc.get_dims(), tensor.options());
  if (context.weight_is_channels_last_) {
    if (context.original_desc_.get_ndims() == 4) {
      result = result.to(at::MemoryFormat::ChannelsLast);
    } else if (context.original_desc_.get_ndims() == 5) {
      result = result.to(at::MemoryFormat::ChannelsLast3d);
    }
  }
  ideep::tensor pub_tensor;
  auto pub_tensor_desc = context.original_desc_.to_type(dtype);
  if (ideep::data_type::f32 == dtype) {
    pub_tensor.init(pub_tensor_desc, result.template data_ptr<float>());
  } else {
    pub_tensor.init(pub_tensor_desc, result.template data_ptr<c10::BFloat16>());
  }
  pub_tensor.feed_from(blocked_tensor);
  return result;
}

} // namespace convolution
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
