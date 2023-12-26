#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/csrc/autograd/custom_function.h>

namespace torch_ipex {
namespace cpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_forward(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    bool training,
    double eps,
    std::array<bool, 3> grad_input_mask);

class IPEXInstanceNormOp
    : public torch::autograd::Function<IPEXInstanceNormOp> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const c10::optional<at::Tensor>& weight_opt,
      const c10::optional<at::Tensor>& bias_opt,
      const c10::optional<at::Tensor>& running_mean_opt,
      const c10::optional<at::Tensor>& running_var_opt,
      bool training,
      double momentum,
      double eps);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

using instancenorm_forward_fn = std::vector<at::Tensor> (*)(
    const at::Tensor& /* input */,
    const at::Tensor& /* weight */,
    const at::Tensor& /* bias */,
    const at::Tensor& /* running mean*/,
    const at::Tensor& /* running var*/,
    double eps,
    bool is_channels_last);

using instancenorm_backward_fn = std::vector<at::Tensor> (*)(
    const at::Tensor& /* dY */,
    const at::Tensor& /* X */,
    const at::Tensor& /* weight */,
    const at::Tensor& /* mean */,
    const at::Tensor& /* variance */,
    bool is_channels_last);

IPEX_DECLARE_DISPATCH(instancenorm_forward_fn, InstanceNormKernel);
IPEX_DECLARE_DISPATCH(instancenorm_backward_fn, InstanceNormBackwardKernel);

} // namespace cpu
} // namespace torch_ipex
