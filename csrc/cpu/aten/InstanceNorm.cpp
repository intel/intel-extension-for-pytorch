#include "InstanceNorm.h"
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/record_function.h>
#include <c10/util/accumulate.h>
#include <torch/all.h>
#include "aten/utils/isa_help.h"
#include "autocast/autocast_mode.h"
#include "ideep/IDeepConversions.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(InstanceNormKernel);
IPEX_DEFINE_DISPATCH(InstanceNormBackwardKernel);

namespace {
void check_dims_match_num_input_features(
    const char* arg_name,
    SymInt expected,
    SymInt actual) {
  TORCH_CHECK(
      actual == expected,
      arg_name,
      " should contain ",
      expected,
      " elements not ",
      actual);
}

static inline at::Tensor repeat_if_defined(const at::Tensor& t, SymInt repeat) {
  if (t.defined()) {
    return t.repeat_symint(repeat);
  }
  return t;
}
} // namespace

at::Tensor instance_norm_pytorch(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt /* optional */,
    const c10::optional<at::Tensor>& bias_opt /* optional */,
    const c10::optional<at::Tensor>& running_mean_opt /* optional */,
    const c10::optional<at::Tensor>& running_var_opt /* optional */,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  TORCH_CHECK(
      use_input_stats || (running_mean.defined() && running_var.defined()),
      "Expected running_mean and running_var to be defined when use_input_stats is false");
  std::vector<SymInt> shape = input.sym_sizes().vec();
  SymInt b = input.sym_size(0);
  SymInt c = input.sym_size(1);
  shape[1] = b * c;
  shape[0] = SymInt(1);

  at::Tensor weight_ = repeat_if_defined(weight, b);
  at::Tensor bias_ = repeat_if_defined(bias, b);
  at::Tensor running_mean_ = repeat_if_defined(running_mean, b);
  at::Tensor running_var_ = repeat_if_defined(running_var, b);

  auto input_reshaped = input.contiguous().view_symint(shape);
  auto out = at::batch_norm(
      input_reshaped,
      weight_,
      bias_,
      running_mean_,
      running_var_,
      use_input_stats,
      momentum,
      eps,
      cudnn_enabled);

  // we alias running_mean and running_var because they are const but we want to
  // modify their data
  if (running_mean.defined()) {
    at::alias(running_mean)
        .copy_(running_mean_.view_symint({b, c}).mean(0, false));
  }
  if (running_var.defined()) {
    at::alias(running_var)
        .copy_(running_var_.view_symint({std::move(b), std::move(c)})
                   .mean(0, false));
  }

  return out.view_symint(input.sym_sizes());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_forward(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::native_instance_norm\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::native_instance_norm", c10::ArrayRef<c10::IValue>({}));

  const at::Tensor& weight =
      c10::value_or_else(weight_opt, [] { return at::Tensor(); });
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  bool use_running_stat = (running_mean.defined() && running_var.defined());

  auto num_features = input.sym_sizes()[1];

  if (input.sym_numel() == 0) {
    auto options = input.options().dtype(
        at::toAccumulateType(input.scalar_type(), /*is_cuda=*/input.is_cuda()));
    auto save_mean =
        at::empty_symint(c10::SymIntArrayRef({num_features}), options);
    auto save_invstd = at::empty_symint(
        c10::SymIntArrayRef({std::move(num_features)}), options);

    // don't return view of input, don't return empty tensor because it will
    // break gradient chain
    auto out = input.clone();
    if (weight.defined())
      out = out * weight[0];
    if (bias.defined())
      out = out + bias[0];
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        out, save_mean, save_invstd);
  }

  if (running_mean.defined()) {
    check_dims_match_num_input_features(
        "running_mean", num_features, running_mean.sym_numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features(
        "running_var", num_features, running_var.sym_numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features(
        "weight", num_features, weight.sym_numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features(
        "bias", std::move(num_features), bias.sym_numel());
  }

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;

  if (!training) {
    // TODO: if running_mean.defined()
    auto output_list = InstanceNormKernel(
        input.device().type(),
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        is_channels_last);
    if (output_list.empty()) {
      return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
    }
    return std::make_tuple(output_list[0], output_list[1], output_list[2]);
  }

  auto output_list = InstanceNormKernel(
      input.device().type(),
      input,
      weight,
      bias,
      running_mean,
      running_var,
      eps,
      is_channels_last);
  if (output_list.empty()) {
    return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
  }

  auto output = output_list[0];
  auto saved_mean = output_list[1];
  auto saved_var = output_list[2];

  if (use_running_stat) {
    auto len = input.numel() / running_mean.numel();
    auto new_mean = (1 - momentum) * running_mean + momentum * saved_mean;
    auto new_var =
        (1 - momentum) * running_var + (momentum * len / (len - 1)) * saved_var;
    return std::make_tuple(output, new_mean, new_var);
  }
  return std::make_tuple(output, saved_mean, saved_var);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    bool training,
    double eps,
    std::array<bool, 3> grad_input_mask) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::native_instance_norm_backward\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::native_instance_norm_backward",
      c10::ArrayRef<c10::IValue>({}));

  if (input.numel() == 0) {
    std::vector<int64_t> dims(input.dim() - 1);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    // don't return empty tensor because it will break gradient chain
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (grad_input_mask[2]) {
      grad_bias = grad_output.sum(dims);
    }
    if (grad_input_mask[1]) {
      grad_weight = (grad_output * input).sum(dims);
    }
    if (grad_input_mask[0] && weight.defined()) {
      grad_input = grad_output * weight[0];
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  auto is_contiguous_any =
      grad_output.is_contiguous(at::MemoryFormat::Contiguous) ||
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  auto grad_output_contiguous = is_contiguous_any
      ? grad_output
      : grad_output.contiguous(input.suggest_memory_format());
  bool is_channels_last = grad_output_contiguous.suggest_memory_format() ==
          at::MemoryFormat::ChannelsLast ||
      grad_output_contiguous.suggest_memory_format() ==
          at::MemoryFormat::ChannelsLast3d;

  auto res = InstanceNormBackwardKernel(
      input.device().type(),
      grad_output_contiguous,
      input,
      weight,
      save_mean,
      save_var,
      is_channels_last);

  auto grad_input = res[0];
  auto grad_weight = res[1];
  auto grad_bias = res[2];

  return std::make_tuple(
      grad_input_mask[0] ? grad_input : at::Tensor(),
      grad_input_mask[1] ? grad_weight : at::Tensor(),
      grad_input_mask[2] ? grad_bias : at::Tensor());
}

at::Tensor IPEXInstanceNormOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool use_input_stats,
    double momentum,
    double eps) {
  RECORD_FUNCTION(
      "IPEXInstanceNormOp::forward", c10::ArrayRef<c10::IValue>({}));

  bool training = use_input_stats;

  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  TORCH_CHECK(
      use_input_stats || (running_mean.defined() && running_var.defined()),
      "Expected running_mean and running_var to be defined when use_input_stats is false");

  ctx->saved_data["train"] = training;
  ctx->saved_data["eps"] = eps;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] = bias.requires_grad();
  at::Tensor output, save_mean, save_var;
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::instance_norm_forward", "")
          .typed<decltype(instance_norm_forward)>();
  std::tie(output, save_mean, save_var) = op.call(
      input,
      weight,
      bias,
      running_mean_opt,
      running_var_opt,
      training,
      momentum,
      eps);
  ctx->save_for_backward({input, weight, save_mean, save_var});
  return output;
}

torch::autograd::variable_list IPEXInstanceNormOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  RECORD_FUNCTION(
      "IPEXInstanceNormOp::backward", c10::ArrayRef<c10::IValue>({}));

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
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::instance_norm_backward", "")
          .typed<decltype(instance_norm_backward)>();
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

at::Tensor instance_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool use_input_stats,
    double momentum,
    double eps,
    bool /* cudnn_enabled, deprecated */) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::instance_norm\n");
#endif
  RECORD_FUNCTION("torch_ipex::instance_norm", c10::ArrayRef<c10::IValue>({}));

  at::Tensor output;
  auto isa = get_current_isa_level();
  if (isa == "AVX2")
    output = instance_norm_pytorch(
                 input,
                 weight_opt,
                 bias_opt,
                 running_mean_opt,
                 running_var_opt,
                 use_input_stats,
                 momentum,
                 eps,
                 false)
                 .contiguous(input.suggest_memory_format());
  else
    output = IPEXInstanceNormOp::apply(
        input,
        weight_opt,
        bias_opt,
        running_mean_opt,
        running_var_opt,
        use_input_stats,
        momentum,
        eps);
  return output;
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "instance_norm_forward(Tensor input, Tensor? weight, Tensor? bias, Tensor? "
      "running_mean, Tensor? running_var, bool train, float momentum, float "
      "eps) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "instance_norm_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::instance_norm_forward);
  m.def(
      "instance_norm_backward(Tensor grad_output, Tensor input, Tensor weight, "
      "Tensor save_mean, Tensor save_var, bool train, float eps, bool[3] "
      "grad_input_mask) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "instance_norm_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::instance_norm_backward);
}

// IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
//   m.impl(
//       TORCH_SELECTIVE_NAME("aten::instance_norm"),
//       TORCH_FN((&torch_ipex::cpu::instance_norm)));
// }

} // namespace cpu
} // namespace torch_ipex