#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
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

Tensor tensordot(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2) {
  TORCH_CHECK(
      dims1.size() == dims2.size(),
      "both dimension lists should have same length");
  int64_t csize = 1; // total size of the contracted dimensions
  Tensor t1 = input1;
  Tensor t2 = input2;
  for (const auto i : c10::irange(dims1.size())) {
    int s1 = input1.size(dims1[i]);
    int s2 = input2.size(dims2[i]);
    if (s2 == 1) { // broadcasted dimensions can be summed right away
      t1 = t1.sum(dims1[i], true);
    } else if (s1 == 1) {
      t2 = t2.sum(dims2[i], true);
    } else {
      TORCH_CHECK(
          s1 == s2,
          "contracted dimensions need to match, but first has size ",
          s1,
          " in dim ",
          dims1[i],
          " and second has size ",
          s2,
          " in dim ",
          dims2[i]);
      csize *= s1;
    }
  }
  auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
  auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
  std::vector<int64_t> p1, p2,
      rsizes; // p1, p2: input permutations, rsizes: sizes of the result
  p1.reserve(input1.dim());
  p2.reserve(input2.dim());
  rsizes.reserve(input1.dim() + input2.dim() - (int64_t)dims1.size());
  int64_t size1 = 1; // number of non-contracted elements in input1
  int64_t size2 = 1; // number of non-contracted elements in input2

  // fill the permutations and compute sizes
  for (const auto i : c10::irange(input1.dim())) {
    if (!cdims1[i]) {
      p1.emplace_back(i);
      size1 *= t1.size(i);
      rsizes.emplace_back(t1.size(i));
    }
  }
  for (const auto x : dims1) {
    p1.emplace_back(x);
  }
  for (const auto x : dims2) {
    p2.emplace_back(x);
  }
  for (const auto i : c10::irange(input2.dim())) {
    if (!cdims2[i]) {
      p2.emplace_back(i);
      size2 *= t2.size(i);
      rsizes.emplace_back(t2.size(i));
    }
  }
  // permut and reshape for matrix multiplication
  t1 = t1.permute(p1).reshape({size1, csize});
  t2 = t2.permute(p2).reshape({csize, size2});
  // multiply and reshape to target size
  return at::mm(t1, t2).reshape(rsizes);
}

Tensor& tensordot_out(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2,
    Tensor& result) {
  Tensor result_tmp = at::tensordot(input1, input2, dims1, dims2);
  auto result_dtype = result_tmp.scalar_type();
  auto output_tensor_dtype = result.scalar_type();
  auto output_device = result.device();
  auto input1_device = input1.device();
  auto input2_device = input2.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
      (output_device == input1_device) && (input1_device == input2_device),
      "tensordot: Expected the output and input tensors to be on the "
      "same device, but got the output tensor on ",
      output_device,
      ", input tensor a on ",
      input1_device,
      ", and input tensor b on ",
      input2_device);
  // check if the computed result has the same dtype as the out tensor
  // (because tensordot does not support type promotion)
  TORCH_CHECK(
      result_dtype == output_tensor_dtype,
      "tensordot",
      ": Expected the output tensor to have dtype ",
      result_dtype,
      ", but got an output tensor with dtype ",
      output_tensor_dtype);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
