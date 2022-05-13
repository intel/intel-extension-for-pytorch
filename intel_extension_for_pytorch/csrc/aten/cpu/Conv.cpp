#include "Conv.h"
#include <torch/extension.h>
#include "WeightPack.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/utils/ipex_op_profile.h"
#include "utils/utils.h"

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
    const ideep::tensor& mkldnn_bias,
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
  const ideep::tensor mkldnn_input_ = itensor_view_from_dense(input);
  ideep::tensor mkldnn_input = mkldnn_input_;
  // The following code forces the 3D input to channels last, which is a
  // temporary workaround before channels last 1D is formally supported in
  // PyTorch.
  if (mkldnn_input_.ndims() == 3 &&
      !mkldnn_input_.get_desc().is_channels_last()) {
    ideep::tensor mkldnn_input_conv1d{
        mkldnn_input_.get_desc().to_format(ideep::format_tag::nwc)};
    mkldnn_input_conv1d.feed_from(mkldnn_input_);
    mkldnn_input = mkldnn_input_conv1d;
  }
  auto output_sizes = output.sizes();

  ideep::tensor mkldnn_output = itensor_view_from_dense(output);

  if (mkldnn_bias.is_empty()) {
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
  } else {
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
  }
}

at::Tensor convolution_kernel(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& mkldnn_bias,
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

  at::Tensor output;
  if (input.dim() != 3) {
    output = at::empty(
        output_sizes,
        input.options().memory_format(input.suggest_memory_format()));
  } else {
    // This a temporary workaround before channels last 1D is formally supported
    // in PyTorch. We will force to return nwc output.
    std::vector<int64_t> output_strides = {
        (output_sizes[1] * output_sizes[2]), 1, output_sizes[1]};
    output = at::empty_strided(output_sizes, output_strides, input.options());
  }

  convolution_kernel_output(
      input,
      mkldnn_weight,
      mkldnn_bias,
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
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_forward_impl\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::convolution_forward_impl", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t());
}

at::Tensor convolution_backward_input(
    at::IntArrayRef input_size,
    const at::Tensor& grad_output,
    const ideep::tensor& mkldnn_weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    bool bias_defined,
    bool weight_use_channels_last) {
  TORCH_CHECK(
      input_size.size() == 4 || input_size.size() == 5,
      "Only support 2d or 3d convolution for convolution_backward_input");

  const ideep::tensor mkldnn_grad_output = itensor_view_from_dense(grad_output);
  bool is_channels_last_contiguous =
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  auto memory_format = at::MemoryFormat::Contiguous;
  if (is_channels_last_contiguous) {
    if (input_size.size() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }

  auto grad_input =
      at::empty(input_size, grad_output.options().memory_format(memory_format));
  ideep::tensor mkldnn_grad_input;
  if (is_channels_last_contiguous) {
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

  if (is_channels_last_contiguous) {
    return grad_input;
  } else {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(mkldnn_grad_input),
        optTypeMetaToScalarType(grad_output.options().dtype_opt()),
        grad_output.options().device_opt()));
  }
}

std::tuple<at::Tensor, at::Tensor> convolution_backward_weights(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const ideep::tensor::desc& packed_weight_desc,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    bool bias_defined) {
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_backward_weights");
  const ideep::tensor mkldnn_grad_output = itensor_view_from_dense(grad_output);
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input);

  bool is_channels_last_contiguous =
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  auto grad_weight = at::empty_like(weight, grad_output.options());
  at::Tensor grad_bias;
  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  if (grad_output.scalar_type() == at::ScalarType::Float) {
    mkldnn_grad_weight.init(
        packed_weight_desc, grad_weight.template data_ptr<float>());
  } else {
    mkldnn_grad_weight.init(
        packed_weight_desc, grad_weight.template data_ptr<c10::BFloat16>());
  }

  std::vector<int64_t> real_weight_size = {
      grad_output.size(1), input.size(1) / groups};
  for (auto& k : kernel_size) {
    real_weight_size.push_back(k);
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
  return std::make_tuple(grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_kernel(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& at_weight,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& mkldnn_bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    const bool weight_channels_last,
    std::array<bool, 3> output_mask) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipeIPEX_RECORD_FUNCTIONx::convolution_backward\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::convolution_backward", std::vector<c10::IValue>({}));

  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_backward");

  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      weight_channels_last;

  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto grad_output_ = grad_output.contiguous(memory_format);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = convolution_backward_input(
        input.sizes(),
        grad_output_,
        mkldnn_weight,
        padding,
        stride,
        dilation,
        kernel_size,
        groups,
        output_mask[2],
        weight_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    auto input_ = input.contiguous(memory_format);
    std::tie(grad_weight, grad_bias) = convolution_backward_weights(
        grad_output_,
        input_,
        at_weight,
        mkldnn_weight.get_desc(),
        padding,
        stride,
        dilation,
        kernel_size,
        groups,
        output_mask[2]);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor IPEXConvolutionOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "IPEXConvolutionOp::_forward", std::vector<c10::IValue>({}));

  return convolution_forward_impl(input, op_context);
}

at::Tensor IPEXConvolutionOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "IPEXConvolutionOp::forward", std::vector<c10::IValue>({}));

  ctx->saved_data["op_context"] = op_context;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias_opt.has_value() && bias_opt.value().requires_grad() ? true : false;
  ctx->save_for_backward({input});

  return convolution_forward_impl(input, op_context);
}

torch::autograd::variable_list IPEXConvolutionOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  IPEX_RECORD_FUNCTION(
      "IPEXConvolutionOp::backward", std::vector<c10::IValue>({}));

  auto op_context =
      ctx->saved_data["op_context"].toCustomClass<ConvolutionOpContext>();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) =
      op_context->run_backward(input, grad_outputs[0], output_mask);
  return {grad_input, grad_weight, grad_bias, at::Tensor()};
}

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context) {
  if (at::GradMode::is_enabled()) {
    return IPEXConvolutionOp::apply(input, weight, bias_opt, op_context);
  }
  return IPEXConvolutionOp::_forward(input, weight, bias_opt, op_context);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "convolution_forward(Tensor input, Tensor weight, Tensor? bias, "
      "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor",
      torch_ipex::cpu::convolution_forward);
}

} // namespace

namespace torch_ipex {
namespace autocast {

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::intrusive_ptr<torch_ipex::cpu::ConvolutionOpContext>&
        op_context) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::convolution_forward", "")
                       .typed<decltype(convolution_forward)>();
  auto target_type = get_autocast_dtype();

  // TODO: make check weight dtype should be float for training case.
  return op.call(
      cpu_cached_cast(target_type, input), weight, bias_opt, op_context);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("convolution_forward", torch_ipex::autocast::convolution_forward);
}

} // namespace autocast
} // namespace torch_ipex
