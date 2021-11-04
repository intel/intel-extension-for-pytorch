#include "ConvTranspose.h"
#include <torch/extension.h>
#include "WeightPack.h"
#include "mkldnn/MKLDNNCommon.h"
#include "mkldnn/MKLDNNConversions.h"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include "torch_ipex/csrc/utils.h"

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

at::Tensor conv_transpose2d_kernel_impl(
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
  const ideep::tensor x = itensor_from_tensor(input);
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, input.suggest_memory_format());
    y = itensor_from_tensor(output);
  }
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        b,
        output_sizes,
        y,
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups);
  } else {
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        output_sizes,
        y,
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups);
  }

  if (!is_channels_last) {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(y),
        optTypeMetaToScalarType(input.options().dtype_opt()),
        input.options().device_opt()));
  } else {
    TORCH_INTERNAL_ASSERT(y.get_desc().is_nhwc());
    return output;
  }
}

at::Tensor convolution_transpose_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked) {
  ideep::tensor w;
  std::vector<int64_t> origin_weight_dims;
  if (weight_prepacked) {
    origin_weight_dims.push_back(input.size(1));
    origin_weight_dims.push_back(output_channel / groups);
    for (auto& s : kernel_size) {
      origin_weight_dims.push_back(s);
    }

    w = get_conv_transpose2d_packed_weight(
        weight,
        stride,
        padding,
        dilation,
        origin_weight_dims,
        groups,
        weight_channels_last,
        weight_prepacked,
        weight_channels_last,
        {},
        ideep::attr_t());
  } else {
    for (auto& s : weight.sizes()) {
      origin_weight_dims.push_back(s);
    }

    w = itensor_from_tensor(weight);
    // mkldnn transposed convolution has weight in logical order of OIHW or
    // OIDHW, while PyTorch has IOHW or IODHW, `._tranpose()` switches strides
    // (no memory copy).
    w.transpose_(0, 1);
  }

  return conv_transpose2d_kernel_impl(
      input,
      w,
      bias_opt,
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      origin_weight_dims,
      ideep::attr_t());
}

at::Tensor conv_transpose2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked) {
  return convolution_transpose_impl(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      kernel_size,
      output_channel,
      weight_channels_last,
      weight_prepacked);
}

at::Tensor IPEXConvTransposeOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked) {
  at::AutoNonVariableTypeMode g;
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "IPEXConvTransposeOp::_forward", std::vector<c10::IValue>({}));
#endif

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::conv_transpose2d", "")
                       .typed<decltype(conv_transpose2d)>();
  return op.call(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      kernel_size,
      output_channel,
      weight_channels_last,
      weight_prepacked);
}

at::Tensor IPEXConvTransposeOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXConvTransposeOp::forward", std::vector<c10::IValue>({}));
#endif
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["groups"] = groups;
  ctx->saved_data["output_padding"] = output_padding;
  ctx->saved_data["weight_channels_last"] = weight_channels_last;
  ctx->saved_data["weight_prepacked"] = weight_prepacked;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias_opt.has_value() && bias_opt.value().requires_grad() ? true : false;
  ctx->save_for_backward({input, weight});

  return _forward(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      kernel_size,
      output_channel,
      weight_channels_last,
      weight_prepacked);
}

at::Tensor conv_transpose2d_backward_input(
    at::IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    bool bias_defined,
    bool weight_channels_last,
    bool weight_prepacked) {
  auto grad_input = at::empty({0}, grad_output.options());
  bool is_channels_last =
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  auto grad_y = itensor_from_tensor(grad_output);

  std::vector<int64_t> origin_weight_dims;
  origin_weight_dims.push_back(input_size[1]);
  origin_weight_dims.push_back(grad_output.size(1) / groups);
  for (auto& s : kernel_size) {
    origin_weight_dims.push_back(s);
  }

  ideep::tensor w;
  if (weight_prepacked) {
    w = get_conv_transpose2d_packed_weight(
        weight,
        stride,
        padding,
        dilation,
        origin_weight_dims,
        groups,
        weight_channels_last,
        weight_prepacked,
        weight_channels_last,
        {},
        ideep::attr_t());
  } else {
    w = itensor_view_from_dense(weight).transpose_(0, 1);
  }

  ideep::tensor grad_x;
  if (is_channels_last) {
    grad_input.resize_(input_size, grad_output.suggest_memory_format());
    grad_x = itensor_from_tensor(grad_input);
  }

  ideep::convolution_transpose_backward_data::compute(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      padding.vec(),
      padding.vec(),
      dilation.vec(),
      groups);

  if (!is_channels_last) {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(grad_x),
        optTypeMetaToScalarType(grad_output.options().dtype_opt()),
        grad_output.options().device_opt()));
  } else {
    TORCH_INTERNAL_ASSERT(grad_x.get_desc().is_nhwc());
    return grad_input;
  }
}

std::tuple<at::Tensor, at::Tensor> conv_transpose2d_backward_weights(
    at::IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    bool bias_defined,
    bool weight_use_channels_last,
    bool weight_prepacked) {
  bool is_channels_last =
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  auto grad_y = itensor_from_tensor(grad_output);
  auto x = itensor_from_tensor(input);

  auto grad_weight = at::empty(weight_size, grad_output.options());
  at::Tensor grad_bias;
  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  std::vector<int64_t> real_weight_size = {
      input.size(1), grad_output.size(1) / groups};
  for (auto& k : kernel_size) {
    real_weight_size.push_back(k);
  }

  if (weight_prepacked) {
    // weight has been packed, mkldnn_grad_weight shares buffer with
    // grad_weight;
    mkldnn_grad_weight = get_conv_transpose2d_packed_weight(
        grad_weight,
        stride,
        padding,
        dilation,
        real_weight_size,
        groups,
        weight_use_channels_last,
        weight_prepacked,
        weight_use_channels_last,
        {},
        ideep::attr_t());
  }

  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    mkldnn_grad_bias = itensor_view_from_dense(grad_bias);
    ideep::convolution_transpose_backward_weights::compute(
        x,
        grad_y,
        {real_weight_size.begin(), real_weight_size.end()},
        mkldnn_grad_weight,
        mkldnn_grad_bias,
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups);
  } else {
    ideep::convolution_transpose_backward_weights::compute(
        x,
        grad_y,
        {real_weight_size.begin(), real_weight_size.end()},
        mkldnn_grad_weight,
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups);
  }

  if (weight_prepacked) {
    // weight has been packed, mkldnn_grad_weight shares buffer with
    // grad_weight;
    return std::make_tuple(grad_weight, grad_bias);
  }
  if (!is_channels_last) {
    return std::make_tuple(
        mkldnn_to_dense(new_with_itensor_mkldnn(
            std::move(mkldnn_grad_weight),
            optTypeMetaToScalarType(grad_output.options().dtype_opt()),
            grad_output.options().device_opt())),
        grad_bias);
  } else {
    return std::make_tuple(
        mkldnn_to_dense(
            new_with_itensor_mkldnn(
                std::move(mkldnn_grad_weight),
                optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                grad_output.options().device_opt()))
            .to(at::MemoryFormat::ChannelsLast),
        grad_bias);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_transpose2d_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    std::array<bool, 3> output_mask,
    bool weight_channels_last,
    bool weight_prepacked) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::conv_transpose2d_backward\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::conv_transpose2d_backward", std::vector<c10::IValue>({}));
#endif
  auto memory_format = input.suggest_memory_format();
  at::Tensor grad_output = grad_output_t.contiguous(memory_format);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = conv_transpose2d_backward_input(
        input.sizes(),
        grad_output,
        weight,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        kernel_size,
        output_mask[2],
        weight_channels_last,
        weight_prepacked);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = conv_transpose2d_backward_weights(
        weight.sizes(),
        grad_output,
        input,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        kernel_size,
        output_mask[2],
        weight_channels_last,
        weight_prepacked);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

torch::autograd::variable_list IPEXConvTransposeOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "IPEXConvTransposeOp::backward", std::vector<c10::IValue>({}));
#endif
  auto stride = ctx->saved_data["stride"].toIntVector();
  auto padding = ctx->saved_data["padding"].toIntVector();
  auto output_padding = ctx->saved_data["output_padding"].toIntVector();
  auto groups = ctx->saved_data["groups"].toInt();
  auto dilation = ctx->saved_data["dilation"].toIntVector();
  auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
  auto weight_channels_last = ctx->saved_data["weight_channels_last"].toBool();
  auto weight_prepacked = ctx->saved_data["weight_prepacked"].toBool();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = conv_transpose2d_backward(
      input,
      grad_outputs[0],
      weight,
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      kernel_size,
      output_mask,
      weight_channels_last,
      weight_prepacked);
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
      at::Tensor(),
      at::Tensor()};
}

at::Tensor conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked) {
  if (at::GradMode::is_enabled()) {
    return IPEXConvTransposeOp::apply(
        input,
        weight,
        bias_opt,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        kernel_size,
        output_channel,
        weight_channels_last,
        weight_prepacked);
  }
  return IPEXConvTransposeOp::_forward(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      kernel_size,
      output_channel,
      weight_channels_last,
      weight_prepacked);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::conv_transpose2d", "")
                       .typed<decltype(conv_transpose2d)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("conv_transpose2d");
#endif
  auto target_type = get_autocast_dtype();

  // TODO: make check weight dtype should be float for training case.
  return op.call(
      cpu_cached_cast(target_type, input),
      cpu_cached_cast(target_type, weight),
      cpu_cached_cast(target_type, bias_opt),
      stride,
      padding,
      output_padding,
      groups,
      dilation,
      kernel_size,
      output_channel,
      weight_channels_last,
      weight_prepacked);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "conv_transpose2d(Tensor input, Tensor weight, Tensor? bias_opt, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation, int[] kernel_size, int output_channel, bool weight_channels_last, bool weight_prepacked) -> Tensor");
  m.impl(
      "conv_transpose2d",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::conv_transpose2d);
  m.impl(
      "conv_transpose2d",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::conv_transpose2d_forward);
  m.impl(
      "conv_transpose2d",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::conv_transpose2d);
  m.def(
      "conv_transpose2d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation, int[] kernel_size, bool[3] output_mask, bool weight_channels_last, bool weight_prepacked) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "conv_transpose2d_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::conv_transpose2d_backward);
}

} // namespace
