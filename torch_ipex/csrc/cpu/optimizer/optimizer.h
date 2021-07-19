#pragma once
#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu{
std::tuple<at::Tensor, at::Tensor, at::Tensor> lamb_fused_step(
    const at::Tensor& param_,
    const at::Tensor& exp_avg_,
    const at::Tensor& exp_avg_sq_,
    const at::Tensor& grad_,
    const at::Tensor& param2_,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps);
std::tuple<at::Tensor, at::Tensor> adagrad_fused_step(
    const at::Tensor& param_,
    const at::Tensor& grad_,
    const at::Tensor& state_sum_,
    const at::Tensor& param2_,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps);
void packed_add
    (at::Tensor &top_half,
      at::Tensor &bot_half,
      const at::Tensor &grad,
      double alpha);
} // namespace cpu
} // namespace torch_ipex
