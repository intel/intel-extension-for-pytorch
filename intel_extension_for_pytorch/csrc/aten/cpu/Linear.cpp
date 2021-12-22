#include <torch/extension.h>

#include "Eltwise.h"
#include "Linear.h"
#include "WeightPack.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/utils/utils.h"

namespace torch_ipex {
namespace cpu {

enum EltwiseType { NotFused = 0, ReLU = 1, Sigmoid = 2 };
/**
 * Linear inplace version with oneDNN kernel.
 * Inplace version will be used when user provides output tensor. eg: Linear+Add
 *fusion.
 *
 *
 *@param self Activatin input for Linear
 *@param weight Weight for Linear
 *@param bias Bias for Linear
 *@param output Output tensor provided by user
 *@param attr Attribute for oneDNN primitive.
 */
void linear_kernel_output(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const ideep::attr_t& attr) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  const ideep::tensor mkldnn_input = itensor_view_from_dense(self_reshaped);
  auto output_size = output.sizes();
  auto output_memory_format = output.suggest_memory_format();
  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {
        self_reshaped.size(0), mkldnn_weight.get_dim(0)};
    output = output.reshape(output_size_reshaped);
  }
  ideep::tensor mkldnn_output = itensor_view_from_dense(output);

  if (bias.defined()) {
    auto bias_ = self.is_contiguous() ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = itensor_view_from_dense(bias_);
    ideep::inner_product_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        mkldnn_output,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::inner_product_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_output,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }
  if (self.dim() != 2) {
    output = output.reshape(output_size);
    output = output.to(output_memory_format);
  }
}

at::Tensor linear_kernel(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(mkldnn_weight.get_dim(0));
  auto output = at::empty(output_size, self.options());
  linear_kernel_output(self, mkldnn_weight, bias, output, attr);
  return output;
}

at::Tensor linear_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_ = *bias_maybe_owned;
  const ideep::tensor mkldnn_weight =
      get_linear_packed_weight(weight, out_features, in_features);
  return linear_kernel(self, mkldnn_weight, bias_, ideep::attr_t());
}

at::Tensor linear_eltwise_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_ = *bias_maybe_owned;
  const ideep::tensor mkldnn_weight =
      get_linear_packed_weight(weight, out_features, in_features);
  auto attr = ideep::attr_t();
  if (eltwise == ReLU)
    attr = ideep::attr_t::fuse_relu();
  else
    attr = ideep::attr_t::fuse_sigmoid();
  return linear_kernel(self, mkldnn_weight, bias_, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    std::array<bool, 3> output_mask) {
  at::Tensor grad_input, grad_weight, grad_bias;
  // weight's desc is needed for both bw_d and bw_w
  const ideep::tensor w =
      get_linear_packed_weight(weight, out_features, in_features);
  // for IP, currently both stag=ab and dtag=ab are only supported by onednn, we
  // need first make both src and diff_dst contiguous if the input or
  // grad_output is not expected
  auto input_contiguous = input.is_contiguous() ? input : input.contiguous();
  auto input_reshaped = input_contiguous.dim() > 2
      ? input_contiguous.reshape(
            {-1, input_contiguous.size(input_contiguous.dim() - 1)})
      : input_contiguous;
  auto grad_output_contiguous =
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  auto grad_output_reshaped = grad_output_contiguous.dim() > 2
      ? grad_output_contiguous.reshape(
            {-1, grad_output_contiguous.size(grad_output_contiguous.dim() - 1)})
      : grad_output_contiguous;
  const ideep::tensor grady = itensor_view_from_dense(grad_output_reshaped);
  if (output_mask[0]) {
    at::Tensor grad_input_reshaped = at::empty_like(input_reshaped);
    ideep::tensor gradx = itensor_view_from_dense(grad_input_reshaped);

    // bw_d
    ideep::inner_product_backward_data::compute(
        grady, w, input_reshaped.sizes().vec(), gradx);
    grad_input = input_contiguous.dim() > 2
        ? grad_input_reshaped.reshape(input_contiguous.sizes().vec())
        : grad_input_reshaped;
  }
  if (output_mask[1] || output_mask[2]) {
    // bw_w
    grad_weight = at::empty_like(weight);
    const ideep::tensor x = itensor_view_from_dense(input_reshaped);
    auto diff_weight_type = w.get_data_type();
    ideep::tensor gradw(w.get_desc(), grad_weight.data_ptr());
    if (output_mask[2]) {
      grad_bias = at::empty({w.get_dim(0)}, weight.options());
      ideep::tensor gradb = itensor_view_from_dense(grad_bias);
      ideep::inner_product_backward_weights::compute(
          x, grady, gradw, gradb, diff_weight_type);
    } else {
      ideep::inner_product_backward_weights::compute(
          x, grady, gradw, diff_weight_type);
    }
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor IPEXLinearOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise) {
  at::AutoNonVariableTypeMode g;
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLinearOp::_forward", std::vector<c10::IValue>({}));
#endif
  if (eltwise == NotFused) {
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_ipex::ipex_linear", "")
                         .typed<decltype(ipex_linear)>();
    return op.call(input, weight, out_features, in_features, bias);
  } else {
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("torch_ipex::ipex_linear_eltwise", "")
            .typed<decltype(ipex_linear_eltwise)>();
    return op.call(input, weight, out_features, in_features, bias, eltwise);
  }
}

at::Tensor IPEXLinearOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLinearOp::forward", std::vector<c10::IValue>({}));
#endif
  at::AutoNonVariableTypeMode g;
  ctx->saved_data["out_features"] = out_features;
  ctx->saved_data["in_features"] = in_features;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias.has_value() && bias.value().requires_grad() ? true : false;
  ctx->saved_data["eltwise"] = eltwise;
  auto output =
      _forward(input, weight, out_features, in_features, bias, eltwise);
  if (eltwise == NotFused)
    ctx->save_for_backward({input, weight});
  else
    ctx->save_for_backward({input, weight, output});
  return output;
}

torch::autograd::tensor_list IPEXLinearOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLinearOp::backward", std::vector<c10::IValue>({}));
#endif
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  int64_t out_features = ctx->saved_data["out_features"].toInt();
  int64_t in_features = ctx->saved_data["in_features"].toInt();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  int64_t eltwise = ctx->saved_data["eltwise"].toInt();
  at::Tensor grad_output;
  if (eltwise == NotFused) {
    grad_output = grad_outputs[0];
  } else {
    at::Tensor output = saved[2];
    grad_output = eltwise == ReLU
        ? relu_use_dst_for_bwd(grad_outputs[0], output)
        : sigmoid_use_dst_for_bwd(grad_outputs[0], output);
  }

  at::Tensor grad_input, grad_weight, grad_bias;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::linear_backward", "")
                       .typed<decltype(linear_backward)>();
  std::tie(grad_input, grad_weight, grad_bias) = op.call(
      input, grad_output, weight, out_features, in_features, output_mask);
  // must have save nums of output with inputs args
  return {
      grad_input,
      grad_weight,
      at::Tensor(),
      at::Tensor(),
      grad_bias,
      at::Tensor()};
}

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias) {
  if (at::GradMode::is_enabled())
    return IPEXLinearOp::apply(input, weight, out_features, in_features, bias);
  return IPEXLinearOp::_forward(input, weight, out_features, in_features, bias);
}

at::Tensor ipex_linear_eltwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise) {
  return IPEXLinearOp::apply(
      input, weight, out_features, in_features, bias, eltwise);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_linear", "")
                       .typed<decltype(ipex_linear)>();
  auto target_type = get_autocast_dtype();
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16 ||
          weight.scalar_type() == at::kFloat,
      "ipex_linear only support bfloat16 and float autocast dtype");
  return op.call(
      cpu_cached_cast(target_type, input),
      cpu_cached_cast(target_type, weight),
      out_features,
      in_features,
      cpu_cached_cast(target_type, bias));
}

at::Tensor ipex_linear_eltwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_linear_eltwise", "")
                       .typed<decltype(ipex_linear_eltwise)>();
  auto target_type = get_autocast_dtype();
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16 ||
          weight.scalar_type() == at::kFloat,
      "ipex_linear_eltwise only support bfloat16 and float autocast dtype");
  return op.call(
      cpu_cached_cast(target_type, input),
      cpu_cached_cast(target_type, weight),
      out_features,
      in_features,
      cpu_cached_cast(target_type, bias),
      eltwise);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "ipex_linear(Tensor input, Tensor weight, int out_features, int in_features, Tensor? bias) -> Tensor");
  m.impl(
      "ipex_linear", c10::DispatchKey::Autograd, torch_ipex::cpu::ipex_linear);
  m.impl(
      "ipex_linear",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::ipex_linear);
  m.impl("ipex_linear", c10::DispatchKey::CPU, torch_ipex::cpu::linear_forward);
  // fuse eltwise
  m.def(
      "ipex_linear_eltwise(Tensor input, Tensor weight, int out_features, int in_features, Tensor? bias, int eltwise) -> Tensor");
  m.impl(
      "ipex_linear_eltwise",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::ipex_linear_eltwise);
  m.impl(
      "ipex_linear_eltwise",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::ipex_linear_eltwise);
  m.impl(
      "ipex_linear_eltwise",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::linear_eltwise_forward);
  // bw
  m.def(
      "linear_backward(Tensor input, Tensor grad_output, Tensor weight, int out_features, int in_features, bool[3] out_mask) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "linear_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::linear_backward);
}

} // namespace
