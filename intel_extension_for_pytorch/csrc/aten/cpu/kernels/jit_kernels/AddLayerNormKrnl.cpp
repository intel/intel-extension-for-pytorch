// this file is main from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/layer_norm_kernel.cpp
//  and
//  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp

#include <csrc/jit/cpu/kernels/AddLayerNorm.h>
#include "csrc/utils/ipex_op_profile.h"

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/add_layernorm.h"
#endif
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor add_layer_norm_kernel_impl(
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

#if defined(DYN_DISP_BUILD)
} // anonymous namespace

REGISTER_DISPATCH(add_layer_norm_kernel_stub, &add_layer_norm_kernel_impl);

#endif

} // namespace cpu
} // namespace torch_ipex