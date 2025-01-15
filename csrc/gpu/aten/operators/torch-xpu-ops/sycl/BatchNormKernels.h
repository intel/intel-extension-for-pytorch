#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor, Tensor>
batch_norm_backward_reduce_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g);

TORCH_XPU_API std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double epsilon,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean_opt,
    const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask);
} // namespace xpu
} // namespace native
} // namespace at
