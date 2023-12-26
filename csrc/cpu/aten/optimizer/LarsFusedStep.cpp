#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#include <cmath>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(lars_norm_kernel_stub);

/**
 * LARS fused update kernel.
 * Support Double, Float, BFloat16 training
 *@param param_ Parameters to be update
 *@param grad_ Grad used to update Parameters
 *@param momentum_buf_ momentum to accelerate convergence
 *@param param2_ Used for BF16 training, if param_ is float, param2_ is bf16
 *params need to be synced after update if param_ is BFloat16, param2_ is
 *params_ last 16 bit matissa to construct float params
 *@param momentum Args for momentum.
 *@param learning_rate  Weight for grad while update.
 *@param eeta Trust coefficient
 *@param eps Prevent division by zero
 *@param weight_decay Args for regularization to avoid over-fit.
 *@param dampening Attribute for momentum.
 *@param nesterov Attribute for momentum.
 */
c10::optional<at::Tensor> lars_fused_step(
    at::Tensor& param_,
    const at::Tensor& grad_,
    const c10::optional<at::Tensor>& momentum_buf_,
    at::Tensor& param2_,
    double momentum,
    double learning_rate,
    double eeta,
    double eps,
    double weight_decay,
    double dampening,
    bool nesterov) {
  RECORD_FUNCTION(
      "torch_ipex::lars_fused_step", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad_ sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      !momentum_buf_.has_value() ||
          param_.sizes() == momentum_buf_.value().sizes(),
      "Expect param and momentum_buf have the same sizes, param sizes: ",
      param_.sizes(),
      "; momentum_buf sizes: ",
      momentum_buf_.value().sizes());
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  at::Tensor grad_f32 = grad_.to(torch::kFloat32);
  float w_norm = lars_norm_kernel_stub(kCPU, param_);
  float g_norm = lars_norm_kernel_stub(kCPU, grad_f32);

  float trust_ratio = 1.f;
  if ((w_norm > 0) && (g_norm > 0)) {
    trust_ratio = eeta * w_norm / (g_norm + weight_decay * w_norm + eps);
  }
  learning_rate *= trust_ratio;
  return sgd_fused_step_kernel_stub(
      kCPU,
      param_,
      grad_,
      momentum_buf_,
      param2_,
      momentum,
      learning_rate,
      weight_decay,
      dampening,
      nesterov);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "lars_fused_step(Tensor param, Tensor grad, Tensor? momentum_buf, Tensor "
      "trail, float momentum, float learning_rate, float eeta, float eps,"
      "float weight_decay, float dampening, bool nesterov) -> Tensor?",
      torch_ipex::cpu::lars_fused_step);
}

} // namespace
