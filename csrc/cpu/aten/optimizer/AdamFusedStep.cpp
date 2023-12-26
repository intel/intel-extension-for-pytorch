#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(adam_fused_step_kernel_stub);

void adam_fused_step(
    const at::Tensor& param_,
    const at::Tensor& exp_avg_,
    const at::Tensor& exp_avg_sq_,
    const at::Tensor& max_exp_avg_sq_,
    const at::Tensor& grad_,
    const at::Tensor& param2_,
    bool amsgrad,
    double step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  RECORD_FUNCTION(
      "torch_ipex::adam_fused_step", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      learning_rate >= 0, "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(eps >= 0, "Expect eps >= 0.0, got ", eps);
  TORCH_CHECK(beta1 >= 0 && beta1 < 1, "Expect 0.0 <= beta1 < 1.0, got", beta1);
  TORCH_CHECK(beta2 >= 0 && beta2 < 1, "Expect 0.0 <= beta2 < 1.0, got", beta2);
  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      param_.sizes() == exp_avg_.sizes(),
      "Expect param and exp_avg have the same sizes, param sizes: ",
      param_.sizes(),
      "; exp_avg sizes: ",
      exp_avg_.sizes());
  TORCH_CHECK(
      param_.sizes() == exp_avg_sq_.sizes(),
      "Expect param and exp_avg_sq_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; exp_avg_sq sizes: ",
      exp_avg_sq_.sizes());
  if (amsgrad) {
    TORCH_CHECK(
        param_.sizes() == max_exp_avg_sq_.sizes(),
        "Expect param and max_exp_avg_sq_ have the same sizes, param sizes: ",
        param_.sizes(),
        "; max_exp_avg_sq sizes: ",
        max_exp_avg_sq_.sizes());
  }
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  /*
  pointer to adam_fused_step_kernel_impl(
      param_,
      exp_avg_,
      exp_avg_sq_,
      max_exp_avg_sq_,
      grad_,
      param2_,
      amsgrad,
      step,
      beta1,
      beta2,
      learning_rate,
      weight_decay,
      eps);
  */
  adam_fused_step_kernel_stub(
      kCPU,
      param_,
      exp_avg_,
      exp_avg_sq_,
      max_exp_avg_sq_,
      grad_,
      param2_,
      amsgrad,
      step,
      beta1,
      beta2,
      learning_rate,
      weight_decay,
      eps);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "adam_fused_step",
      torch_ipex::cpu::adam_fused_step,
      at::DispatchKey::CPU);
}

} // namespace
