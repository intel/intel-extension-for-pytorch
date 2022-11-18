#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
inline std::pair<int64_t, int64_t> _check_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  return std::make_pair(M, N);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor output, mean, rstd;
  if (input.numel() == 0) {
    output = at::empty({M, N}, input.options());
    mean = at::empty({M}, input.options());
    rstd = at::empty({M}, input.options());
  } else {
    Tensor input_compatible =
        xpu::oneDNN::is_broadcast(input) ? input.contiguous() : input;
    Tensor weight_compatible =
        weight.defined() && xpu::oneDNN::is_broadcast(weight)
        ? weight.contiguous()
        : weight;
    Tensor bias_compatible = bias.defined() && xpu::oneDNN::is_broadcast(bias)
        ? bias.contiguous()
        : bias;
    if (input.dim() == 1)
      input_compatible = input_compatible.reshape({M, N});
    if (weight.defined() && weight.dim() == 1)
      weight_compatible = weight_compatible.reshape({N});
    if (bias.defined() && bias.dim() == 1)
      bias_compatible = bias_compatible.reshape({N});

    // inteprete memory as {1, M, N} due to same physical layout
    if (!(input_compatible.size(0) == M && input_compatible.size(1) == N)) {
      std::tie(output, mean, rstd) = xpu::oneDNN::layer_norm(
          input_compatible.reshape({1, M, N}),
          weight_compatible,
          bias_compatible,
          epsilon);
    } else {
      // by pass reshape (reorder)
      std::tie(output, mean, rstd) = xpu::oneDNN::layer_norm(
          input_compatible, weight_compatible, bias_compatible, epsilon);
    }
  }
  return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0 || grad_output.numel() == 0) {
    grad_input = at::empty(input.sizes(), input.options());
    if (weight.numel() != 0) {
      grad_weight = at::zeros(weight.sizes(), weight.options());
      grad_bias = at::zeros(bias.sizes(), bias.options());
    } else {
      grad_weight = at::empty(weight.sizes(), weight.options());
      grad_bias = at::empty(bias.sizes(), bias.options());
    }
  } else {
    Tensor input_compatible =
        xpu::oneDNN::is_broadcast(input) ? input.contiguous() : input;
    Tensor grad_output_compatible = xpu::oneDNN::is_broadcast(grad_output)
        ? grad_output.contiguous()
        : grad_output;
    Tensor weight_compatible =
        weight.defined() && xpu::oneDNN::is_broadcast(weight)
        ? weight.contiguous()
        : weight;
    if (input.dim() == 1)
      input_compatible = input_compatible.reshape({M, N});
    if (grad_output.dim() == 1)
      grad_output_compatible = grad_output_compatible.reshape({M, N});
    if (weight.defined() && weight.dim() == 1)
      weight_compatible = weight_compatible.reshape({N});

    // inteprete memory as {1, M, N} due to same physical layout
    if (!(input_compatible.size(0) == M && input_compatible.size(1) == N)) {
      std::tie(grad_input, grad_weight, grad_bias) =
          xpu::oneDNN::layer_norm_backward(
              grad_output_compatible.reshape({{1, M, N}}),
              input_compatible.reshape({1, M, N}),
              mean,
              rstd,
              weight_compatible,
              1e-5);
    } else {
      std::tie(grad_input, grad_weight, grad_bias) =
          xpu::oneDNN::layer_norm_backward(
              grad_output_compatible,
              input_compatible,
              mean,
              rstd,
              weight_compatible,
              1e-5);
    }
  }
  return std::make_tuple(
      grad_input.reshape(input.sizes()),
      weight.defined() ? grad_weight.reshape(weight.sizes()) : grad_weight,
      bias.defined() ? grad_bias.reshape(bias.sizes()) : grad_bias);
}

} // namespace AtenIpexTypeXPU
} // namespace at
