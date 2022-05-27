// this file is main from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/layer_norm_kernel.cpp
//  and
//  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp

#include "AddLayerNorm.h"
#include "csrc/utils/ipex_op_profile.h"

#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(add_layer_norm_kernel_stub);

at::Tensor AddLayerNorm(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps) {
  /*
  pointer to add_layer_norm_kernel_impl(
      a, b, alpha, normalized_shape, weight_opt, bias_opt, eps);
  */
  return add_layer_norm_kernel_stub(
      kCPU, a, b, alpha, normalized_shape, weight_opt, bias_opt, eps);
}

at::Tensor dil_add_layernorm(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps,
    bool cuda_enable) {
  IPEX_RECORD_FUNCTION("dil_add_layernorm", c10::ArrayRef<c10::IValue>({}));

  // no broadcast
  bool no_broadcast = true;
  for (auto i = 0; i < a.ndimension(); i++) {
    if (a.size(i) != b.size(i)) {
      no_broadcast = false;
      break;
    }
  }
  // Only support 64byte aligned
  bool aligned_64_bytes = a.size(a.ndimension() - 1) % 16 == 0 &&
      b.size(b.ndimension() - 1) % 16 == 0;
  // Only support contiguous tensor
  bool is_contiguous = a.is_contiguous() && b.is_contiguous();
  if (no_broadcast && aligned_64_bytes && is_contiguous && alpha == 1.0f) {
    return AddLayerNorm(
        a, b, alpha, normalized_shape, weight_opt, bias_opt, eps);
  } else {
    auto add_res = at::add(a, b, alpha);
    return at::layer_norm(add_res, normalized_shape, weight_opt, bias_opt, eps);
  }
}
} // namespace cpu
} // namespace torch_ipex
