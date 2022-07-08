#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {


class IPEXBatchNormOp : public torch::autograd::Function<IPEXBatchNormOp> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const at::Tensor& bias,
      const c10::optional<at::Tensor>& running_mean_opt,
      const c10::optional<at::Tensor>& running_var_opt,
      bool train,
      double momentum,
      double eps);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

} // namespace cpu
} // namespace torch_ipex
