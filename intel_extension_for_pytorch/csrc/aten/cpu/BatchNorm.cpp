#include <torch/extension.h>

#include "BatchNorm.h"
#include "csrc/cpu/ideep/IDeepConversions.h"

#include "csrc/autocast/autocast_mode.h"
#include "csrc/autocast/autocast_verbose.h"

namespace torch_ipex {
namespace cpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  const at::Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  ideep::tensor x = itensor_view_from_dense(input);
  ideep::tensor w = itensor_view_from_dense(weight);
  ideep::tensor b = itensor_view_from_dense(bias);
  bool use_running_stat = (running_mean.defined() && running_var.defined());

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto output = at::empty(
      input.sizes(),
      input.options().memory_format(input.suggest_memory_format()));
  ideep::tensor y;
  if (is_channels_last) {
    y = itensor_view_from_dense(output);
  }
  if (train) {
    // TODO: enable 3d batchnorm.
    TORCH_CHECK(
        input.dim() == 4,
        "batch_norm: currently mkldnn training only support 2d batchnorm");
    auto saved_mean = at::empty(input.size(1), weight.options());
    auto saved_var = at::empty(input.size(1), weight.options());
    ideep::tensor mkldnn_saved_mean = itensor_view_from_dense(saved_mean);
    ideep::tensor mkldnn_saved_var = itensor_view_from_dense(saved_var);
    ideep::batch_normalization_forward_training::compute(
        x, w, b, y, mkldnn_saved_mean, mkldnn_saved_var, momentum, eps);
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      ideep::tensor m = itensor_view_from_dense(running_mean);
      ideep::tensor v = itensor_view_from_dense(running_var);
      const std::vector<float> scales_mean{
          static_cast<float>(1 - momentum), static_cast<float>(momentum)};
      const std::vector<float> scales_var{
          static_cast<float>(1 - momentum),
          static_cast<float>(momentum * len / (len - 1))};
      ideep::sum::compute(scales_mean, {m, mkldnn_saved_mean}, m);
      ideep::sum::compute(scales_var, {v, mkldnn_saved_var}, v);
    }
    if (is_channels_last) {
      return std::make_tuple(output, saved_mean, saved_var);
    } else {
      return std::make_tuple(
          mkldnn_to_dense(new_with_itensor_mkldnn(
              std::move(y),
              optTypeMetaToScalarType(input.options().dtype_opt()),
              input.options().device_opt())),
          saved_mean,
          saved_var);
    }
  } else {
    TORCH_CHECK(
        input.dim() == 4 || input.dim() == 5,
        "batch_norm: currently mkldnn inference only support 2d and 3d "
        "batchnorm");
    if (use_running_stat) {
      ideep::tensor m = itensor_view_from_dense(running_mean);
      ideep::tensor v = itensor_view_from_dense(running_var);
      ideep::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps);
    } else {
      // TODO: keep running estimates.
      TORCH_CHECK(
          false,
          "mkldnn_batch_norm: mkldnn inference is not keep running estimates.");
    }

    if (is_channels_last) {
      return std::make_tuple(output, running_mean, running_var);
    } else {
      return std::make_tuple(
          mkldnn_to_dense(new_with_itensor_mkldnn(
              std::move(y),
              optTypeMetaToScalarType(input.options().dtype_opt()),
              input.options().device_opt())),
          running_mean,
          running_var);
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask) {
  ideep::tensor grady = itensor_view_from_dense(grad_output);
  ideep::tensor x = itensor_view_from_dense(input);
  ideep::tensor w = itensor_view_from_dense(weight);
  ideep::tensor m = itensor_view_from_dense(save_mean);
  ideep::tensor v = itensor_view_from_dense(save_var);

  bool is_channels_last =
      grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto grad_input = at::empty(
      grad_output.sizes(),
      grad_output.options().memory_format(grad_output.suggest_memory_format()));
  auto grad_weight = at::empty(grad_output.size(1), weight.options());
  auto grad_bias = at::empty(grad_output.size(1), weight.options());
  ideep::tensor gradx, gradw, gradb;
  if (is_channels_last) {
    gradx = itensor_view_from_dense(grad_input);
  }
  gradw = itensor_view_from_dense(grad_weight);
  gradb = itensor_view_from_dense(grad_bias);
  if (train) {
    ideep::batch_normalization_backward::compute(
        x, m, v, grady, w, gradx, gradw, gradb, eps);
  } else {
    ideep::batch_normalization_backward::compute(
        x,
        m,
        v,
        grady,
        w,
        gradx,
        gradw,
        gradb,
        eps,
        ideep::tensor(),
        ideep::batch_normalization_flag::use_global_stats);
  }

  if (is_channels_last) {
    return std::make_tuple(
        grad_input_mask[0] ? grad_input : at::Tensor(),
        grad_input_mask[1] ? grad_weight : at::Tensor(),
        grad_input_mask[2] ? grad_bias : at::Tensor());
  } else {
    return std::make_tuple(
        grad_input_mask[0]
            ? mkldnn_to_dense(new_with_itensor_mkldnn(
                  std::move(gradx),
                  optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                  grad_output.options().device_opt()))
            : at::Tensor(),
        grad_input_mask[1] ? grad_weight : at::Tensor(),
        grad_input_mask[2] ? grad_bias : at::Tensor());
  }
}

at::Tensor IPEXBatchNormOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXBatchNormOp::forward", std::vector<c10::IValue>({}));
#endif
  ctx->saved_data["train"] = train;
  ctx->saved_data["eps"] = eps;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] = bias.requires_grad();
  at::Tensor output, save_mean, save_var;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::batch_norm_forward", "")
                       .typed<decltype(batch_norm_forward)>();
  std::tie(output, save_mean, save_var) = op.call(
      input,
      weight,
      bias,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      eps);
  ctx->save_for_backward({input, weight, save_mean, save_var});
  return output;
}

torch::autograd::variable_list IPEXBatchNormOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXBatchNormOp::backward", std::vector<c10::IValue>({}));
#endif
  auto train = ctx->saved_data["train"].toBool();
  auto eps = ctx->saved_data["eps"].toDouble();

  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  at::Tensor save_mean = saved[2];
  at::Tensor save_var = saved[3];
  at::Tensor grad_input, grad_weight, grad_bias;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::batch_norm_backward", "")
                       .typed<decltype(batch_norm_backward)>();
  std::tie(grad_input, grad_weight, grad_bias) = op.call(
      grad_outputs[0],
      input,
      weight,
      save_mean,
      save_var,
      train,
      eps,
      output_mask);
  return {
      grad_input,
      grad_weight,
      grad_bias,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps,
    bool cudnn_enabled) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::batch_norm", std::vector<c10::IValue>({}));
#endif
  // Only 2d bfloat16 training calling onednn path, and this path will be
  // discarded after aten batchnorm optimized well.
  if (weight_opt.has_value() && weight_opt.value().defined() &&
      bias_opt.has_value() && bias_opt.value().defined() &&
      !torch::jit::tracer::isTracing() && input.ndimension() == 4 && train &&
      input.scalar_type() == at::kBFloat16 &&
      weight_opt.value().scalar_type() == at::kFloat) {
    return IPEXBatchNormOp::apply(
        input,
        weight_opt.value(),
        bias_opt.value(),
        running_mean_opt,
        running_var_opt,
        train,
        momentum,
        eps);
  } else {
    return at::batch_norm(
        input,
        weight_opt,
        bias_opt,
        running_mean_opt,
        running_var_opt,
        train,
        momentum,
        eps,
        cudnn_enabled);
  }
}

at::Tensor frozen_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::frozen_batch_norm", std::vector<c10::IValue>({}));
#endif
  return IPEXBatchNormOp::apply(
      input, weight, bias, running_mean, running_var, false, 0, 0);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "frozen_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor "
      "running_mean, Tensor running_var) -> Tensor");
  m.impl(
      "frozen_batch_norm",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::frozen_batch_norm);
  m.def(
      "batch_norm_forward(Tensor input, Tensor weight, Tensor bias, Tensor? "
      "running_mean, Tensor? running_var, bool train, float momentum, float "
      "eps) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "batch_norm_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::batch_norm_forward);
  m.def(
      "batch_norm_backward(Tensor grad_output, Tensor input, Tensor weight, "
      "Tensor save_mean, Tensor save_var, bool train, float eps, bool[3] "
      "grad_input_mask) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "batch_norm_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::batch_norm_backward);
}

} // namespace

namespace torch_ipex {
namespace autocast {

at::Tensor frozen_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::frozen_batch_norm", "")
                       .typed<decltype(frozen_batch_norm)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("frozen_batch_norm");
#endif
  return op.call(input, weight, bias, running_mean, running_var);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("frozen_batch_norm", torch_ipex::autocast::frozen_batch_norm);
}

} // namespace autocast
} // namespace torch_ipex
