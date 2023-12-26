#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(sgd_fused_step_kernel_stub);

/**
 * SGD fused update kernel.
 * Support Double, Float, BFloat16 training
 *@param param_ Parameters to be update
 *@param grad_ Grad used to update Parameters
 *@param momentum_buf_ momentum to accelerate convergence
 *@param param2_ Used for BF16 training, if param_ is float, param2_ is bf16
 *params need to be synced after update if param_ is BFloat16, param2_ is
 *params_ last 16 bit matissa to construct float params
 *@param momentum Args for momentum.
 *@param learning_rate  Weight for grad while update.
 *@param weight_decay Args for regularization to avoid over-fit.
 *@param dampening Attribute for momentum.
 *@param nesterov Attribute for momentum.
 */
c10::optional<at::Tensor> sgd_fused_step(
    at::Tensor& param_,
    const at::Tensor& grad_,
    const c10::optional<at::Tensor>& momentum_buf_,
    at::Tensor& param2_,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov) {
  RECORD_FUNCTION("torch_ipex::sgd_fused_step", c10::ArrayRef<c10::IValue>({}));

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

  /*
  pointer to sgd_fused_step_kernel_impl(
      param_,
      grad_,
      momentum_buf_,
      param2_,
      momentum,
      learning_rate,
      weight_decay,
      dampening,
      nesterov);
  */
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
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "sgd_fused_step", torch_ipex::cpu::sgd_fused_step, at::DispatchKey::CPU);
}
} // namespace
