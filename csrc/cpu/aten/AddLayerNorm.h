#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

using Tensor = at::Tensor;
using IntArrayRef = at::IntArrayRef;
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

/**
 * This operator fuse add + layernorm
 * */
at::Tensor AddLayerNorm(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps);

at::Tensor dil_add_layernorm(
    const at::Tensor& input,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps,
    bool cuda_enable);

namespace {

at::Tensor add_layer_norm_kernel_impl(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps);
}

using add_layer_norm_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    int,
    at::IntArrayRef,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    float);
IPEX_DECLARE_DISPATCH(add_layer_norm_kernel_fn, add_layer_norm_kernel_stub);

} // namespace cpu
} // namespace torch_ipex