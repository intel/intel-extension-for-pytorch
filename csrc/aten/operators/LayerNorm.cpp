#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_layer_norm_inputs(
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

  if (!(input.size(0) == M && input.size(1) == N)) {
    Tensor output, mean, rstd;
    std::tie(output, mean, rstd) = xpu::oneDNN::layer_norm(
        input.reshape({1, M, N}), weight, bias, epsilon);
    return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
  } else {
    // by pass reshape (reorder)
    return xpu::oneDNN::layer_norm(input, weight, bias, epsilon);
  }
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
  if (!(input.size(0) == M && input.size(1) == N)) {
    std::tie(grad_input, grad_weight, grad_bias) =
        xpu::oneDNN::layer_norm_backward(
            grad_output.reshape({{1, M, N}}),
            input.reshape({1, M, N}),
            mean,
            rstd,
            weight,
            1e-5);
    return std::make_tuple(
        grad_input.reshape(input.sizes()), grad_weight, grad_bias);
  } else {
    return xpu::oneDNN::layer_norm_backward(
        grad_output, input, mean, rstd, weight, 1e-5);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
