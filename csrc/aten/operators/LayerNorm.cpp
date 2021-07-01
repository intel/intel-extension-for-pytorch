#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    double epsilon) {
  if (!(input.size(0) == M && input.size(1) == N)) {
    Tensor output, mean, rstd;
    std::tie(output, mean, rstd) = xpu::oneDNN::layer_norm(
        input.view({1, M, N}), weight, bias, epsilon);
    return std::make_tuple(output.view(input.sizes()), mean, rstd);
  } else {
    // by pass reshape (reorder)
    return xpu::oneDNN::layer_norm(input, weight, bias, epsilon);
  }
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (!(input.size(0) == M && input.size(1) == N)) {
    std::tie(grad_input, grad_weight, grad_bias) =
        xpu::oneDNN::layer_norm_backward(grad_output.view({{1, M, N}}),
            input.view({1, M, N}), mean, rstd, weight, 1e-5);
    return std::make_tuple(
        grad_input.view(input.sizes()), grad_weight, grad_bias);
  } else {
    return xpu::oneDNN::layer_norm_backward(
        grad_output, input, mean, rstd, weight, 1e-5);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
