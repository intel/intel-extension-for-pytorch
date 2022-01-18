// this file is main from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/layer_norm_kernel.cpp
//  and
//  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp

#include "AddLayerNorm.h"

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/add_layernorm.h"
#endif
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace kernels {

at::Tensor AddLayerNorm(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps) {
#if defined(CPU_CAPABILITY_AVX512)
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(a, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = a.contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  if (a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat) {
    torch_ipex::cpu::kernel::vec::vec512::AddLayerNormKernelImpl<float, float>(
        X, b, alpha, weight, bias, M, N, eps, Y);
  } else if (
      a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16) {
    if (weight.defined() && weight.scalar_type() == at::kBFloat16) {
      torch_ipex::cpu::kernel::vec::vec512::
          AddLayerNormKernelImpl<at::BFloat16, at::BFloat16>(
              X, b, alpha, weight, bias, M, N, eps, Y);
    } else {
      torch_ipex::cpu::kernel::vec::vec512::
          AddLayerNormKernelImpl<at::BFloat16, float>(
              X, b, alpha, weight, bias, M, N, eps, Y);
    }
  }
  return Y;
#else
  return at::layer_norm(
      at::add(a, b, alpha), normalized_shape, weight_opt, bias_opt, eps);
#endif
}

} // namespace kernels
} // namespace cpu
} // namespace jit
} // namespace torch_ipex

namespace torch_ipex {
namespace cpu {
at::Tensor dil_add_layernorm(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps,
    bool cuda_enable) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_add_layernorm", std::vector<c10::IValue>({}));
#endif
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
    return jit::cpu::kernels::AddLayerNorm(
        a, b, alpha, normalized_shape, weight_opt, bias_opt, eps);
  } else {
    auto add_res = at::add(a, b, alpha);
    return at::layer_norm(add_res, normalized_shape, weight_opt, bias_opt, eps);
  }
}
} // namespace cpu
} // namespace torch_ipex
