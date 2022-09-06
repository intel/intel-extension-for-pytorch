#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <core/TensorImplUtils.h>
#include <intrinsic/intrinsic.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>

#include <oneDNN/oneDNN.h>

#include "comm/ParamUtils.h"
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_gelu", std::vector<c10::IValue>({input, weight, bias}));
  auto result = at::empty({0}, input.options());
  Tensor _bias = bias.defined() ? bias : at::Tensor();
  if (input.dim() == 2) {
    // Fused op is marginally faster.
    AtenIpexTypeXPU::matmul(
        result,
        input,
        weight,
        _bias,
        at::Tensor(),
        1.f,
        1.f,
        false,
        xpu::oneDNN::MatmulAttr::kind_with_gelu);
    return result;
  }

  if (input.dim() == 3 && input.is_contiguous()) {
    // Also hit the fused path for contiguous 3D input.
    const auto input_sizes = input.sizes();
    auto input_view =
        input.view({input_sizes[0] * input_sizes[1], input_sizes[2]});
    AtenIpexTypeXPU::matmul(
        result,
        input_view,
        weight,
        _bias,
        at::Tensor(),
        1.f,
        1.f,
        false,
        xpu::oneDNN::MatmulAttr::kind_with_gelu);
    return result.view({input_sizes[0], input_sizes[1], result.size(1)});
  }

  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::gelu(output);
}

Tensor linear_add(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& accumu,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_add", std::vector<c10::IValue>({input, weight, bias}));
  const auto input_sizes = input.sizes();
  const auto weight_sizes = weight.sizes();
  std::vector<int64_t> output_sizes = {
      input_sizes[0], input_sizes[1], weight_sizes[1]};
  if (accumu.sizes().vec() == output_sizes) {
    auto result = at::empty({0}, input.options());
    Tensor _bias = bias.defined() ? bias : at::Tensor();
    if (input.dim() == 2) {
      // Fused op is marginally faster.
      AtenIpexTypeXPU::matmul(
          result,
          input,
          weight,
          _bias,
          accumu,
          1.f,
          1.f,
          false,
          xpu::oneDNN::MatmulAttr::kind_with_sum);
      return result;
    }

    if (input.dim() == 3 && input.is_contiguous()) {
      // Also hit the fused path for contiguous 3D input.
      auto input_view =
          input.view({input_sizes[0] * input_sizes[1], input_sizes[2]});
      auto accumu_view =
          accumu.view({output_sizes[0] * output_sizes[1], output_sizes[2]});
      AtenIpexTypeXPU::matmul(
          result,
          input_view,
          weight,
          _bias,
          accumu_view,
          1.f,
          1.f,
          false,
          xpu::oneDNN::MatmulAttr::kind_with_sum);
      return result.view(output_sizes);
    }

    auto output = at::matmul(input, weight.t());
    if (bias.defined()) {
      output.add_(bias);
    }
    result = at::add(output, accumu);
    return result;
  }

  auto output = at::linear(input, weight, bias);
  auto result = at::add(output, accumu, alpha.to<float>());
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
