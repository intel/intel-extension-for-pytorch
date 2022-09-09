#pragma once
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> lamb_fused_step_kernel_impl(
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

std::tuple<at::Tensor, at::Tensor> adagrad_fused_step_kernel_impl(
    const at::Tensor& param_,
    const at::Tensor& grad_,
    const at::Tensor& state_sum_,
    const at::Tensor& param2_,
    double step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps);

c10::optional<at::Tensor> sgd_fused_step_kernel_impl(
    at::Tensor& param_,
    const at::Tensor& grad_,
    const c10::optional<at::Tensor>& momentum_buf_,
    at::Tensor& param2_,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov);

void packed_add_kernel_impl(
    at::Tensor& top_half,
    at::Tensor& bot_half,
    const at::Tensor& grad,
    double alpha);

void adam_fused_step_kernel_impl(
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
    double eps);

} // namespace

using adagrad_fused_step_kernel_fn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    double,
    double,
    double,
    double);
DECLARE_DISPATCH(adagrad_fused_step_kernel_fn, adagrad_fused_step_kernel_stub);

using lamb_fused_step_kernel_fn =
    std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        int64_t,
        double,
        double,
        double,
        double,
        double);
DECLARE_DISPATCH(lamb_fused_step_kernel_fn, lamb_fused_step_kernel_stub);

using sgd_fused_step_kernel_fn = c10::optional<at::Tensor> (*)(
    at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    at::Tensor&,
    double,
    double,
    double,
    double,
    bool);
DECLARE_DISPATCH(sgd_fused_step_kernel_fn, sgd_fused_step_kernel_stub);

using packed_add_kernel_fn =
    void (*)(at::Tensor&, at::Tensor&, const at::Tensor&, double);
DECLARE_DISPATCH(packed_add_kernel_fn, packed_add_kernel_stub);

using adam_fused_step_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    bool,
    double,
    double,
    double,
    double,
    double,
    double);
DECLARE_DISPATCH(adam_fused_step_kernel_fn, adam_fused_step_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
