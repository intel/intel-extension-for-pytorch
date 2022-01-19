#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

void linear_kernel_output(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const ideep::attr_t& attr);

at::Tensor linear_kernel(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr);

// IPEX customized linear OP with n-D packed weight
// Additional out_features, in_features is used to query expected weigth desc
// Since n-D packed weight have loss these info
class IPEXLinearOp : public torch::autograd::Function<IPEXLinearOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const int64_t out_features,
      const int64_t in_features,
      const c10::optional<at::Tensor>& bias,
      const int64_t eltwise = 0);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const int64_t out_features,
      const int64_t in_features,
      const c10::optional<at::Tensor>& bias,
      const int64_t eltwise = 0);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias);

at::Tensor ipex_linear_eltwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise);

} // namespace cpu
} // namespace torch_ipex
