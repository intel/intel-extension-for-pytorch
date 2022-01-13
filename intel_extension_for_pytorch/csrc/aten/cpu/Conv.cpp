#include "Conv.h"
#include <torch/extension.h>
#include "WeightPack.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/utils/utils.h"

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
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

void convolution_kernel_output(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  // Convolution output kernel, assuming the output always has same format with
  // input, so this function will not change input and output's format, making
  // sure you has made pre-process for input and output to make them have same
  // format before call this function.
  TORCH_CHECK(
      input.suggest_memory_format() == output.suggest_memory_format(),
      "input and output need has same format for convolution_kernel_output");
  TORCH_CHECK(
      (IS_CONTIGUOUS_ANY(input)) && (IS_CONTIGUOUS_ANY(output)),
      "input and output are need contiguous tensor for "
      "convolution_kernel_output");
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input);
  auto output_sizes = output.sizes();

  ideep::tensor mkldnn_output = itensor_view_from_dense(output);

  if (bias.defined()) {
    auto bias_ = IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = itensor_view_from_dense(bias_);
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        {output_sizes.begin(), output_sizes.end()},
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
        {output_sizes.begin(), output_sizes.end()},
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
  // Base convolution kernel, this base kernel will not change input's format,
  // so make sure you has make process the input's format before call this
  // function, the output wil has same format with input.
  // TODO: the input will be actively converted to channels last format
  // after the 5-D tensor supports channels last format.
  TORCH_CHECK(
      IS_CONTIGUOUS_ANY(input),
      "input is need to a contiguous tensor for convolution_kernel");
  auto kernel_size = mkldnn_weight.get_dims();
  auto input_size = input.sizes();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  auto output = at::empty(
      output_sizes,
      input.options().memory_format(input.suggest_memory_format()));
  convolution_kernel_output(
      input,
      mkldnn_weight,
      bias_opt,
      output,
      stride,
      padding,
      dilation,
      groups,
      attr);
  return output;
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
    bool weight_channels_last,
    bool weight_packed,
    const ideep::attr_t& attr) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_forward_impl\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::convolution_forward_impl", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      weight.scalar_type() == input.scalar_type(),
      "the input and weight need have same data type");
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_forward_impl");
  // TODO: add bias dtype check
  // case 1: weight is not packed, check weight.suggest_memory_format()
  // case 2: weight is packed or use user's setting, weight_channels_last.
  bool weight_use_channels_last =
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      weight_channels_last;
  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight_use_channels_last;
  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto input_ = input.to(memory_format);
  at::Tensor weight_ = weight;
  // if weight is not packed, convert format, and weight will has same format
  // with input.
  if (!weight_packed) {
    weight_ = weight_.contiguous(memory_format);
  }
  // get original weight dims.
  std::vector<int64_t> origin_weight_dims;
  origin_weight_dims.push_back(output_channel);
  origin_weight_dims.push_back(input_.size(1) / groups);
  for (auto& s : kernel_size) {
    origin_weight_dims.push_back(s);
  }
  ideep::tensor mkldnn_weight = get_conv_packed_weight(
      weight_,
      stride,
      padding,
      dilation,
      origin_weight_dims,
      groups,
      weight_channels_last,
      weight_packed,
      weight_channels_last,
      {},
      attr);

  return convolution_kernel(
      input_, mkldnn_weight, bias_opt, stride, padding, dilation, groups, attr);
}

at::Tensor convolution_backward_input(
    at::IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    bool bias_defined,
    bool weight_use_channels_last,
    bool weight_packed) {
  TORCH_CHECK(
      input_size.size() == 4 || input_size.size() == 5,
      "Only support 2d or 3d convolution for convolution_backward_input");

  const ideep::tensor mkldnn_grad_output = itensor_view_from_dense(grad_output);
  bool is_channels_last =
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;

  std::vector<int64_t> origin_weight_dims;
  origin_weight_dims.push_back(grad_output.size(1));
  origin_weight_dims.push_back(input_size[1] / groups);
  for (auto& s : kernel_size) {
    origin_weight_dims.push_back(s);
  }
  ideep::tensor mkldnn_weight = get_conv_packed_weight(
      weight,
      stride,
      padding,
      dilation,
      origin_weight_dims,
      groups,
      weight_use_channels_last,
      weight_packed,
      weight_use_channels_last,
      {},
      ideep::attr_t());

  auto grad_input = at::empty(
      input_size,
      grad_output.options().memory_format(grad_output.suggest_memory_format()));
  ideep::tensor mkldnn_grad_input;
  if (is_channels_last) {
    mkldnn_grad_input = itensor_view_from_dense(grad_input);
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
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(mkldnn_grad_input),
        optTypeMetaToScalarType(grad_output.options().dtype_opt()),
        grad_output.options().device_opt()));
  }
}

std::tuple<at::Tensor, at::Tensor> convolution_backward_weights(
    at::IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    bool bias_defined,
    bool weight_use_channels_last,
    bool weight_packed) {
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_backward_weights");
  const ideep::tensor mkldnn_grad_output = itensor_view_from_dense(grad_output);
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input);
  bool is_channels_last =
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;

  auto grad_weight = at::empty(weight_size, grad_output.options());
  at::Tensor grad_bias;
  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  std::vector<int64_t> real_weight_size = {
      grad_output.size(1), input.size(1) / groups};
  for (auto& k : kernel_size) {
    real_weight_size.push_back(k);
  }
  if (weight_packed) {
    // weight has be packed, mkldnn_grad_weight share buffer with
    // grad_weight;
    mkldnn_grad_weight = get_conv_packed_weight(
        grad_weight,
        stride,
        padding,
        dilation,
        real_weight_size,
        groups,
        weight_use_channels_last,
        weight_packed,
        weight_use_channels_last,
        {},
        ideep::attr_t());
  }

  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    mkldnn_grad_bias = itensor_view_from_dense(grad_bias);
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

  if (weight_packed) {
    return std::make_tuple(grad_weight, grad_bias);
  } else {
    if (is_channels_last) {
      auto memory_format = input.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                            : at::MemoryFormat::ChannelsLast3d;
      return std::make_tuple(
          mkldnn_to_dense(
              new_with_itensor_mkldnn(
                  std::move(mkldnn_grad_weight),
                  optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                  grad_output.options().device_opt()))
              .to(memory_format),
          grad_bias);
    } else {
      return std::make_tuple(
          mkldnn_to_dense(new_with_itensor_mkldnn(
              std::move(mkldnn_grad_weight),
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
    std::array<bool, 3> output_mask,
    bool weight_channels_last,
    bool weight_packed) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_backward\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::convolution_backward", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      weight.scalar_type() == input.scalar_type() &&
          weight.scalar_type() == grad_output_t.scalar_type(),
      "the inputs need have same data type");
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_backward");

  bool weight_use_channels_last =
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      weight_channels_last;
  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      weight_use_channels_last;

  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto grad_output_ = grad_output_t.contiguous(memory_format);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    at::Tensor weight_ = weight;
    // if weight is not packed, convert format, and weight will has same format
    // with input.
    if (!weight_packed) {
      weight_ = weight_.contiguous(memory_format);
    }
    grad_input = convolution_backward_input(
        input.sizes(),
        grad_output_,
        weight_,
        padding,
        stride,
        dilation,
        kernel_size,
        groups,
        output_mask[2],
        weight_use_channels_last,
        weight_packed);
  }
  if (output_mask[1] || output_mask[2]) {
    auto input_ = input.contiguous(memory_format);
    std::tie(grad_weight, grad_bias) = convolution_backward_weights(
        weight.sizes(),
        grad_output_,
        input_,
        padding,
        stride,
        dilation,
        kernel_size,
        groups,
        output_mask[2],
        weight_use_channels_last,
        weight_packed);
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
    bool weight_channels_last,
    bool weight_packed) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvolutionOp::_forward", std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      dilation,
      kernel_size,
      groups,
      output_channel,
      weight_channels_last,
      weight_packed,
      ideep::attr_t());
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
    bool weight_channels_last,
    bool weight_packed) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvolutionOp::forward", std::vector<c10::IValue>({}));
#endif
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["groups"] = groups;
  ctx->saved_data["weight_channels_last"] = weight_channels_last;
  ctx->saved_data["weight_packed"] = weight_packed;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias_opt.has_value() && bias_opt.value().requires_grad() ? true : false;
  ctx->save_for_backward({input, weight});

  return convolution_forward_impl(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      dilation,
      kernel_size,
      groups,
      output_channel,
      weight_channels_last,
      weight_packed,
      ideep::attr_t());
}

torch::autograd::variable_list IPEXConvolutionOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvolutionOp::backward", std::vector<c10::IValue>({}));
#endif
  auto stride = ctx->saved_data["stride"].toIntVector();
  auto padding = ctx->saved_data["padding"].toIntVector();
  auto dilation = ctx->saved_data["dilation"].toIntVector();
  auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
  auto groups = ctx->saved_data["groups"].toInt();
  auto weight_channels_last = ctx->saved_data["weight_channels_last"].toBool();
  auto weight_packed = ctx->saved_data["weight_packed"].toBool();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = convolution_backward(
      input,
      grad_outputs[0],
      weight,
      padding,
      stride,
      dilation,
      kernel_size,
      groups,
      output_mask,
      weight_channels_last,
      weight_packed);
  return {
      grad_input,
      grad_weight,
      grad_bias,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
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
    bool weight_channels_last,
    bool weight_packed) {
  if (at::GradMode::is_enabled()) {
    return IPEXConvolutionOp::apply(
        input,
        weight,
        bias_opt,
        stride,
        padding,
        dilation,
        kernel_size,
        groups,
        output_channel,
        weight_channels_last,
        weight_packed);
  }
  return IPEXConvolutionOp::_forward(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      dilation,
      kernel_size,
      groups,
      output_channel,
      weight_channels_last,
      weight_packed);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "convolution_forward(Tensor input, Tensor weight, Tensor? bias, "
      "int[] stride, int[] padding, int[] dilation, int[] kernel_size, int "
      "groups, int output_channel, bool weight_channels_last, bool "
      "weight_packed) -> Tensor",
      torch_ipex::cpu::convolution_forward);
}

} // namespace

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
    bool weight_channels_last,
    bool weight_packed) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::convolution_forward", "")
                       .typed<decltype(convolution_forward)>();
  auto target_type = get_autocast_dtype();

  // TODO: make check weight dtype should be float for training case.
  return op.call(
      cpu_cached_cast(target_type, input),
      cpu_cached_cast(target_type, weight),
      cpu_cached_cast(target_type, bias_opt),
      stride,
      padding,
      dilation,
      kernel_size,
      groups,
      output_channel,
      weight_channels_last,
      weight_packed);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("convolution_forward", torch_ipex::autocast::convolution_forward);
}

} // namespace autocast
} // namespace torch_ipex
