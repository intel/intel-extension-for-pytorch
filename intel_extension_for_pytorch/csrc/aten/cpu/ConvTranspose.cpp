#include "ConvTranspose.h"
#include <torch/extension.h>
#include "WeightPack.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "utils/utils.h"

namespace torch_ipex {
namespace cpu {

constexpr int output_batch_size_dim = 0; // also grad_output
constexpr int weight_input_channels_dim = 1;

std::vector<int64_t> conv_input_size(
    at::IntArrayRef output_size,
    at::IntArrayRef weight_size,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] -
        (2 * padding[d - 2]) + kernel + output_padding[d - 2];
  }
  return input_size;
}

static inline std::vector<int64_t> padding_r(
    at::IntArrayRef padding,
    at::IntArrayRef output_padding) {
  // ConvTranpose padding adjustment
  //
  // PyTorch uses padding/output_padding:
  //   osize = (isize - 1) * stride - 2 * padding + dilation * (kernel_size - 1)
  //   + output_padding + 1
  //
  // MKLDNN uses padding_l/padding_r:
  //   osize = (isize - 1) * stride - padding_l - padding_r + dilation *
  //   (kernel_size - 1) + 1
  //
  // So: padding_l = padding, padding_r = padding - output_padding
  //
  auto dim = padding.size();
  std::vector<int64_t> pad_r(dim);
  for (const auto d : c10::irange(dim)) {
    pad_r[d] = padding[d] - output_padding[d];
  }
  return pad_r;
}

at::Tensor conv_transpose_kernel_impl(
    const at::Tensor& input,
    const ideep::tensor& w,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef origin_weight_dims,
    const ideep::attr_t& attr) {
  std::vector<int64_t> output_sizes = conv_input_size(
      input.sizes(),
      origin_weight_dims,
      padding,
      output_padding,
      stride,
      dilation,
      groups);
  auto output = at::empty({0}, input.options());
  const ideep::tensor x = itensor_view_from_dense(input);
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  const auto memory_format = input.suggest_memory_format();
  bool is_channels_last = memory_format == at::MemoryFormat::ChannelsLast ||
      memory_format == at::MemoryFormat::ChannelsLast3d;

  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_view_from_dense(output);
  }
  if (bias.defined()) {
    const ideep::tensor b = itensor_view_from_dense(bias);
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        b,
        output_sizes,
        y,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        attr);
  } else {
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        output_sizes,
        y,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        attr);
  }

  if (!is_channels_last) {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(y),
        optTypeMetaToScalarType(input.options().dtype_opt()),
        input.options().device_opt()));
  } else {
    return output;
  }
}

void conv_transpose_out_kernel_impl(
    const at::Tensor& input,
    const ideep::tensor& w,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef origin_weight_dims,
    const ideep::attr_t& attr) {
  // ConvTranspose out kernel, assuming the output always has same format with
  // input, so this function will not change input and output's format, making
  // sure you has made pre-process for input and output to make them have same
  // format before call this function.
  TORCH_CHECK(
      input.suggest_memory_format() == output.suggest_memory_format(),
      "input and output need has same format for conv_transpose_out_kernel_impl");
  TORCH_CHECK(
      (IS_CONTIGUOUS_ANY(input)) && (IS_CONTIGUOUS_ANY(output)),
      "input and output should be contiguous tensor for "
      "conv_transpose_out_kernel_impl");
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input);

  std::vector<int64_t> output_sizes = conv_input_size(
      input.sizes(),
      origin_weight_dims,
      padding,
      output_padding,
      stride,
      dilation,
      groups);

  ideep::tensor mkldnn_output = itensor_view_from_dense(output);

  const ideep::tensor x = itensor_view_from_dense(input);
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  if (bias.defined()) {
    const ideep::tensor b = itensor_view_from_dense(bias);
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        b,
        output_sizes,
        mkldnn_output,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        attr);
  } else {
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        output_sizes,
        mkldnn_output,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        attr);
  }
}

at::Tensor conv_transpose_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context) {
  return reinterpret_cast<IpexConvTransposeOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run(input, ideep::attr_t());
}

at::Tensor IPEXConvTransposeOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context) {
  at::AutoNonVariableTypeMode g;
  RECORD_FUNCTION(
      "IPEXConvTransposeOp::_forward", c10::ArrayRef<c10::IValue>({}));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::conv_transpose", "")
                       .typed<decltype(conv_transpose)>();
  return op.call(input, weight, bias_opt, op_context);
}

at::Tensor IPEXConvTransposeOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "IPEXConvTransposeOp::forward", c10::ArrayRef<c10::IValue>({}));

  ctx->saved_data["op_context"] = op_context;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias_opt.has_value() && bias_opt.value().requires_grad() ? true : false;
  ctx->save_for_backward({input});

  return _forward(input, weight, bias_opt, op_context);
}

at::Tensor conv_transpose_backward_input(
    at::IntArrayRef input_size,
    const at::Tensor& grad_output,
    const ideep::tensor& w,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  auto grad_input = at::empty({0}, grad_output.options());
  const auto memory_format = grad_output.suggest_memory_format();
  bool is_channels_last = memory_format == at::MemoryFormat::ChannelsLast ||
      memory_format == at::MemoryFormat::ChannelsLast3d;

  auto grad_y = itensor_view_from_dense(grad_output);

  ideep::tensor grad_x;
  if (is_channels_last) {
    grad_input.resize_(input_size, memory_format);
    grad_x = itensor_view_from_dense(grad_input);
  }

  ideep::convolution_transpose_backward_data::compute(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      padding.vec(),
      padding_r(padding, output_padding),
      dilation.vec(),
      groups);

  if (!is_channels_last) {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(grad_x),
        optTypeMetaToScalarType(grad_output.options().dtype_opt()),
        grad_output.options().device_opt()));
  } else {
    return grad_input;
  }
}

std::tuple<at::Tensor, at::Tensor> conv_transpose_backward_weights(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const ideep::tensor::desc& packed_weight_desc,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    bool bias_defined) {
  auto grad_y = itensor_view_from_dense(grad_output);
  auto x = itensor_view_from_dense(input);

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
      input.size(1), grad_output.size(1) / groups};
  real_weight_size.emplace_back(packed_weight_desc.get_dim(2));
  real_weight_size.emplace_back(packed_weight_desc.get_dim(3));
  if (input.dim() == 5) {
    real_weight_size.emplace_back(packed_weight_desc.get_dim(4));
  }

  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    mkldnn_grad_bias = itensor_view_from_dense(grad_bias);
    ideep::convolution_transpose_backward_weights::compute(
        x,
        grad_y,
        real_weight_size,
        mkldnn_grad_weight,
        mkldnn_grad_bias,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups);
  } else {
    ideep::convolution_transpose_backward_weights::compute(
        x,
        grad_y,
        real_weight_size,
        mkldnn_grad_weight,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups);
  }
  return std::make_tuple(grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    std::array<bool, 3> output_mask,
    const at::Tensor& op_context) {
  return reinterpret_cast<IpexConvTransposeOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_backward(input, grad_output_t, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
conv_transpose_backward_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& at_weight,
    const ideep::tensor& packed_weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    std::array<bool, 3> output_mask,
    bool weight_channels_last) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::conv_transpose_backward\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::conv_transpose_backward", c10::ArrayRef<c10::IValue>({}));

  auto memory_format = input.suggest_memory_format();
  at::Tensor grad_output = grad_output_t.contiguous(memory_format);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = conv_transpose_backward_input(
        input.sizes(),
        grad_output,
        packed_weight,
        stride,
        padding,
        output_padding,
        groups,
        dilation);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = conv_transpose_backward_weights(
        grad_output,
        input,
        at_weight,
        packed_weight.get_desc(),
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        output_mask[2]);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

torch::autograd::variable_list IPEXConvTransposeOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  RECORD_FUNCTION(
      "IPEXConvTransposeOp::backward", c10::ArrayRef<c10::IValue>({}));

  auto op_context = ctx->saved_data["op_context"].toTensor();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) =
      conv_transpose_backward(input, grad_outputs[0], output_mask, op_context);
  return {
      grad_input,
      grad_weight,
      grad_bias,
      at::Tensor()};
}

at::Tensor conv_transpose(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context) {
  if (at::GradMode::is_enabled()) {
    return IPEXConvTransposeOp::apply(input, weight, bias_opt, op_context);
  }
  return IPEXConvTransposeOp::_forward(input, weight, bias_opt, op_context);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor conv_transpose(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::conv_transpose", "")
                       .typed<decltype(conv_transpose)>();
  auto target_type = get_autocast_dtype();

  // TODO: make check weight dtype should be float for training case.
  return op.call(
      cpu_cached_cast(target_type, input), weight, bias_opt, op_context);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "conv_transpose(Tensor input, Tensor weight, Tensor? bias_opt, "
      "Tensor W_prepack) -> Tensor");
  m.impl(
      "conv_transpose",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::conv_transpose);
  m.impl(
      "conv_transpose",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::conv_transpose_forward);
  m.impl(
      "conv_transpose",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::conv_transpose);
  m.def(
      "conv_transpose_backward(Tensor input, Tensor grad_out, bool[3] output_mask, "
      "Tensor W_prepack) "
      " -> (Tensor, Tensor, Tensor)");
  m.impl(
      "conv_transpose_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::conv_transpose_backward);
}

} // namespace
