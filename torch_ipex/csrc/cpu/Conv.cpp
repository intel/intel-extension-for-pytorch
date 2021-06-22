#include <torch/extension.h>
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include "Conv.h"
#include "mkldnn/MKLDNNCommon.h"
#include "torch_ipex/csrc/utils.h"
#include "WeightPrepack.h"

namespace torch_ipex {
namespace cpu {

std::vector<int64_t> calc_conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = kernel_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (kernel_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

at::Tensor convolution_kernel(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  // TODO: the input will be actively converted to channels last format
  // after the 5-D tensor supports channels last format.
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(input);
  auto kernel_size = mkldnn_weight.get_dims();
  std::vector<int64_t> input_size = mkldnn_input.get_dims();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  bool is_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto output = at::empty(output_sizes, input.options().memory_format(input.suggest_memory_format()));
  ideep::tensor mkldnn_output;
  if (is_channels_last) {
    mkldnn_output = at::native::itensor_view_from_dense(output);
  }

  if (bias.defined()) {
    auto bias_ = IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias_);
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }

  if (is_channels_last) {
    return output;
  } else {
    return at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

at::Tensor convolution_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  bool use_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                           weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto mkldnn_memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast
                                                : at::MemoryFormat::Contiguous;
  auto input_ = input.contiguous(mkldnn_memory_format);
  ideep::tensor mkldnn_weight = get_conv_prepacked_weight(input_, weight, stride, padding, dilation, groups, attr, mkldnn_memory_format);
  return convolution_kernel(input_, mkldnn_weight, bias_opt, stride, padding, dilation, groups, attr);
}

void convolution_inplace_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
// TODO: the input will be actively converted to channels last format
// after the 5-D tensor supports channels last format.
  bool use_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                           weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto mkldnn_memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast
                                                : at::MemoryFormat::Contiguous;
  auto input_ = input.contiguous(mkldnn_memory_format);
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(input_);
  ideep::tensor mkldnn_weight = get_conv_prepacked_weight(input, weight, stride, padding, dilation, groups, attr, mkldnn_memory_format);
  auto kernel_size = mkldnn_weight.get_dims();
  std::vector<int64_t> input_size = mkldnn_input.get_dims();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  bool is_channels_last = input_.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  output = IS_CONTIGUOUS_ANY(output) ? output : output.contiguous(output.suggest_memory_format());
  output = output.to(input_.suggest_memory_format());
  ideep::tensor mkldnn_output = at::native::itensor_view_from_dense(output);

  if (bias.defined()) {
    auto bias_ = IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias_);
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }

  if (!is_channels_last) {
    output = at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                            input.options().device_opt()));
  }
}

at::Tensor convolution_forward_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_channels_last) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_forward\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::convolution_forward", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "the input and weight need have same data type");
  // TODO: add bias dtype check
  // case 1: weight is not prepacked, check weight.suggest_memory_format()
  // case 2: weight is prepacked or use user's setting, weight_channels_last.
  bool weight_use_channels_last = weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                                  weight_channels_last;
  bool use_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                           weight_use_channels_last;
  auto mkldnn_memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast
                                                : at::MemoryFormat::Contiguous;
  auto input_ = input.contiguous(mkldnn_memory_format);
  at::Tensor weight_ = weight;
  // if weight is not prepacked, convert format, and weight will has same format with input.
  if (weight_.ndimension() == input.ndimension()) {
    weight_ = weight_.contiguous(mkldnn_memory_format);
  }
  ideep::tensor mkldnn_weight = get_conv_prepacked_weight(weight_, stride, padding, dilation, kernel_size,
      groups, output_channel, /* input_channel */ input_.size(1), weight_use_channels_last);
  return convolution_kernel(input_, mkldnn_weight, bias_opt, stride, padding, dilation, groups, ideep::attr_t());
}

at::Tensor convolution_backward_input(
    at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, at::IntArrayRef kernel_size,
    int64_t groups, bool bias_defined) {
  const ideep::tensor mkldnn_grad_output = at::native::itensor_view_from_dense(grad_output);
  bool is_channels_last = grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  ideep::tensor mkldnn_weight = get_conv_prepacked_weight(weight, stride, padding, dilation, kernel_size,
      groups, grad_output.size(1), input_size[1], is_channels_last);

  auto grad_input = at::empty(input_size, grad_output.options().memory_format(grad_output.suggest_memory_format()));
  ideep::tensor mkldnn_grad_input;
  if (is_channels_last) {
    mkldnn_grad_input = at::native::itensor_view_from_dense(grad_input);
  }

  ideep::convolution_backward_data::compute(
      mkldnn_grad_output,
      mkldnn_weight,
      input_size.vec(),
      mkldnn_grad_input,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);
  
  if (is_channels_last) {
    return grad_input;
  } else {
    return at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_grad_input),
                                            optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                            grad_output.options().device_opt()));
  }
}

std::tuple<at::Tensor, at::Tensor> convolution_backward_weights(
    at::IntArrayRef weight_size , const at::Tensor& grad_output, const at::Tensor& input,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, at::IntArrayRef kernel_size,
    int64_t groups, bool bias_defined) {
  const ideep::tensor mkldnn_grad_output = at::native::itensor_view_from_dense(grad_output);
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(input);
  bool is_channels_last = grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
 
  auto grad_weight = at::empty(weight_size, grad_output.options());
  at::Tensor grad_bias;
  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  if (weight_size.size() != input.ndimension()) {
    // weight has be prepacked, mkldnn_grad_weight share buffer with grad_weight;
    mkldnn_grad_weight = get_conv_prepacked_weight(grad_weight, stride, padding, dilation, kernel_size,
       groups, grad_output.size(1), input.size(1), is_channels_last);
  }
  std::vector<int64_t> real_weight_size = {grad_output.size(1), input.size(1) / groups};
  for (auto& k: kernel_size) {
    real_weight_size.push_back(k);
  }
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    mkldnn_grad_bias = at::native::itensor_view_from_dense(grad_bias);
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        {real_weight_size.begin(), real_weight_size.end()},
        mkldnn_grad_weight,
        mkldnn_grad_bias,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  } else {
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        {real_weight_size.begin(), real_weight_size.end()},
        mkldnn_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  }

  if (weight_size.size() != input.ndimension()) {
    return std::make_tuple(grad_weight, grad_bias);
  } else {
    if (is_channels_last) {
      return std::make_tuple(
          at::native::mkldnn_to_dense(at::native::new_with_itensor_mkldnn(std::move(mkldnn_grad_weight),
                                      optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                      grad_output.options().device_opt())).to(at::MemoryFormat::ChannelsLast),
          grad_bias);
    } else {
      return std::make_tuple(
          at::native::mkldnn_to_dense(at::native::new_with_itensor_mkldnn(std::move(mkldnn_grad_weight),
                                      optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                      grad_output.options().device_opt())),
          grad_bias);
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    std::array<bool,3> output_mask,
    bool weight_channels_last) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_backward\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::convolution_backward", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(weight.scalar_type() == input.scalar_type() && weight.scalar_type() == grad_output_t.scalar_type(),
          "the inputs need have same data type");
  bool weight_use_channels_last = weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                                 weight_channels_last;
  bool use_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                           weight_use_channels_last;

  auto mkldnn_memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast
                                                : at::MemoryFormat::Contiguous;
  auto grad_output_ = grad_output_t.contiguous(mkldnn_memory_format);
  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    at::Tensor weight_ = weight;
    // if weight is not prepacked, convert format, and weight will has same format with input.
    if (weight_.ndimension() == input.ndimension()) {
      weight_ = weight_.contiguous(mkldnn_memory_format);
    }
    grad_input =  convolution_backward_input(input.sizes(), grad_output_, weight_, padding, stride, dilation, kernel_size, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    auto input_ = input.contiguous(mkldnn_memory_format);
    std::tie(grad_weight, grad_bias) = convolution_backward_weights(
         weight.sizes() , grad_output_, input_, padding, stride, dilation, kernel_size, groups, output_mask[2]);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor IPEXConvolutionOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_channels_last){
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvolutionOp::_forward", std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(input, weight, bias_opt, stride, padding, dilation, kernel_size,
      groups, output_channel, weight_channels_last);
}

at::Tensor IPEXConvolutionOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_channels_last){
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvolutionOp::forward", std::vector<c10::IValue>({}));
#endif
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["groups"] = groups;
  // ctx->saved_data["output_channel"] = output_channel;
  ctx->saved_data["weight_channels_last"] = weight_channels_last;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] = bias_opt.has_value() && bias_opt.value().requires_grad() ? true: false;
  ctx->save_for_backward({input, weight});

  return convolution_forward_impl(input, weight, bias_opt, stride, padding, dilation, kernel_size,
      groups, output_channel, weight_channels_last);
}

torch::autograd::variable_list IPEXConvolutionOp::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvolutionOp::backward", std::vector<c10::IValue>({}));
#endif
  auto stride = ctx->saved_data["stride"].toIntVector();
  auto padding = ctx->saved_data["padding"].toIntVector();
  auto dilation = ctx->saved_data["dilation"].toIntVector();
  auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
  auto groups = ctx->saved_data["groups"].toInt();
  // auto output_channel = ctx->saved_data["output_channel"].toInt();
  auto weight_channels_last = ctx->saved_data["weight_channels_last"].toBool();
  std::array<bool,3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1]= ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = convolution_backward(input, grad_outputs[0], weight, padding, stride, dilation,
      kernel_size, groups, output_mask, weight_channels_last);
  return {grad_input, grad_weight, grad_bias, at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_channels_last) {
  if (at::GradMode::is_enabled()) {
    return IPEXConvolutionOp::apply(input, weight, bias_opt, stride, padding, dilation,
        kernel_size, groups, output_channel, weight_channels_last);
  }
  return IPEXConvolutionOp::_forward(input, weight, bias_opt, stride, padding, dilation,
      kernel_size, groups, output_channel, weight_channels_last);
}

}  // namespace cpu
}  // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("convolution_forward(Tensor input, Tensor wieght, Tensor? bias_opt, int[] stride, int[] padding, int[] dilation, int[] kernel_size, int groups, int output_channel, bool weight_channels_last) -> Tensor", torch_ipex::cpu::convolution_forward);
}

}

namespace torch_ipex {
namespace autocast {

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_channels_last) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::convolution_forward", "")
    .typed<decltype(convolution_forward)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("convolution_forward");
#endif
  auto target_type = get_autocast_dtype();

  // TODO: make check weight dtype should be float for training case.
  return op.call(cpu_cached_cast(target_type, input),
                 cpu_cached_cast(target_type, weight),
                 cpu_cached_cast(target_type, bias_opt),
                 stride, padding, dilation, kernel_size, groups, output_channel, weight_channels_last);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("convolution_forward", torch_ipex::autocast::convolution_forward);
}

} // namespace autocast
} // namespace torch_ipex
