#include <torch/extension.h>

#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include "Linear.h"
#include "mkldnn/MKLDNNCommon.h"
#include "WeightPrepack.h"
#include "torch_ipex/csrc/utils.h"

namespace torch_ipex {
namespace cpu {

at::Tensor linear_kernel(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();
  // reshape first if input dim != 2 and the reshape will cost a memory copy.
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(self_reshaped);

  std::vector<int64_t> output_size_reshaped = {self_reshaped.size(0), mkldnn_weight.get_dim(0)};
  auto output = at::empty(output_size_reshaped, self.options());
  ideep::tensor mkldnn_output = at::native::itensor_view_from_dense(output);

  if (bias.defined()) {
    auto bias_ = self.is_contiguous() ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias_);
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

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(mkldnn_weight.get_dim(0));

  if (self.dim() != 2) {
    return output.reshape(output_size);
  }
  return output;
}

at::Tensor linear_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {
  TORCH_CHECK(self.numel() % self.size(self.dim() - 1) == 0);
  const int64_t batch_size = self.numel() / self.size(self.dim() - 1);
  const ideep::tensor mkldnn_weight = get_linear_prepacked_weight(weight, batch_size, self.scalar_type());
  return linear_kernel(self, mkldnn_weight, bias, attr);
}

/**
 * Linear inplace version with oneDNN kernel. 
 * Inplace version will be used when user provides output tensor. eg: Linear+Add fusion. 
 * 
 *
 *@param self Activatin input for Linear
 *@param weight Weight for Linear
 *@param bias Bias for Linear
 *@param output Output tensor provided by user
 *@param attr Attribute for oneDNN primitive.
 *@return output This tensor is provided by user.
 */
at::Tensor linear_inplace_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const ideep::attr_t& attr) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();  
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(self_reshaped);
  const ideep::tensor mkldnn_weight = get_linear_prepacked_weight(weight, mkldnn_input.get_dim(0), self.scalar_type());

  std::vector<int64_t> output_size_reshaped = {self_reshaped.size(0), weight.size(0)};
  output = output.reshape(output_size_reshaped);
  output = output.to(self_.suggest_memory_format());
  ideep::tensor mkldnn_output = at::native::itensor_view_from_dense(output);

  if (bias.defined()) {
    auto bias_ = self.is_contiguous() ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias_);
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

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() != 2) {
    return output.reshape(output_size);
  }
  return output;

}

at::Tensor linear_forward_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {
  const ideep::tensor mkldnn_weight = get_linear_prepacked_weight(weight, out_features, in_features);
  return linear_kernel(self, mkldnn_weight, bias, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward_impl(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    std::array<bool,3> output_mask) {
  at::Tensor grad_input, grad_weight, grad_bias;
  // weight's desc is needed for both bw_d and bw_w
  const ideep::tensor w = get_linear_prepacked_weight(weight, out_features, in_features);
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;
  auto grad_output_reshaped = grad_output.dim() > 2 ? grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  const ideep::tensor grady = at::native::itensor_view_from_dense(grad_output_reshaped);
  if (output_mask[0]) {
    at::Tensor grad_input_reshaped = at::empty_like(input_reshaped);
    ideep::tensor gradx = at::native::itensor_view_from_dense(grad_input_reshaped);

    //bw_d
    ideep::inner_product_backward_data::compute(
      grady, w, input_reshaped.sizes().vec(), gradx
    );
    grad_input = input.dim() > 2 ? grad_input_reshaped.reshape(input.sizes().vec()) : grad_input_reshaped;
  }
  if (output_mask[1] || output_mask[2]) {
  //bw_w
    grad_weight = at::empty_like(weight);
    const ideep::tensor x = at::native::itensor_view_from_dense(input_reshaped);
    auto diff_weight_type = w.get_data_type();
    ideep::tensor gradw(w.get_desc(), grad_weight.data_ptr());
    if (output_mask[2]){
      grad_bias = at::empty({w.get_dim(0)}, weight.options());
      ideep::tensor gradb = at::native::itensor_view_from_dense(grad_bias);
      ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb, diff_weight_type);
    } else {
      ideep::inner_product_backward_weights::compute(x, grady, gradw, diff_weight_type);
    }
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor IPEXLinearOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const at::Tensor& bias){
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLinearOp::_forward", std::vector<c10::IValue>({input, weight, bias}));
#endif
  return linear_forward_impl(input, weight, out_features, in_features, bias, ideep::attr_t());
}

at::Tensor IPEXLinearOp::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const at::Tensor& bias){
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLinearOp::forward", std::vector<c10::IValue>({input, weight, bias}));
#endif
  at::AutoNonVariableTypeMode g;
  ctx->saved_data["input"] = input;
  ctx->saved_data["weight"] = weight;
  ctx->saved_data["out_features"] = out_features;
  ctx->saved_data["in_features"] = in_features;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] = bias.defined() && bias.requires_grad();
  return linear_forward_impl(input, weight, out_features, in_features, bias, ideep::attr_t());
}

torch::autograd::tensor_list IPEXLinearOp::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs){
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLinearOp::backward", std::vector<c10::IValue>({}));
#endif
  auto saved = ctx->get_saved_variables();
  at::Tensor input = ctx->saved_data["input"].toTensor();
  at::Tensor weight = ctx->saved_data["weight"].toTensor();
  int64_t out_features = ctx->saved_data["out_features"].toInt();
  int64_t in_features = ctx->saved_data["in_features"].toInt();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  at::Tensor grad_output = grad_outputs[0];
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = linear_backward_impl(
    input, grad_output, weight, out_features, in_features, output_mask
  );
  // must have save nums of output with inputs args
  return {grad_input, grad_weight, at::Tensor(), at::Tensor(), grad_bias};
}

at::Tensor ipex_linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias_opt) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  // can not pass undefined tensor in "apply" since pytorch have will record the device but
  // undefined tensor dose not have a device
  if (bias.defined()){
    if (at::GradMode::is_enabled())
      return IPEXLinearOp::apply(input, weight, out_features, in_features, bias);
    return IPEXLinearOp::_forward(input, weight, out_features, in_features, bias);
  }
  else {
    if (at::GradMode::is_enabled())
      return IPEXLinearOp::apply(input, weight, out_features, in_features);
    return IPEXLinearOp::_forward(input, weight, out_features, in_features);
  }
}

}  // namespace cpu
}  // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("ipex_linear(Tensor input, Tensor weight, int out_features, int in_features, Tensor? bias_opt) -> Tensor", torch_ipex::cpu::ipex_linear);
}

}

namespace torch_ipex {
namespace autocast {

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias_opt) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::ipex_linear", "")
    .typed<decltype(ipex_linear)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("ipex_linear");
#endif
  auto target_type = get_autocast_dtype();
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kFloat,
        "ipex_linear only support bfloat16 and float autocast dtype");
  return op.call(cpu_cached_cast(target_type, input),
                 cpu_cached_cast(target_type, weight),
                 out_features,
                 in_features,
                 cpu_cached_cast(target_type, bias_opt));
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("ipex_linear", torch_ipex::autocast::ipex_linear);
}

} // namespace autocast
} // namespace torch_ipex

