#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

at::Tensor linear_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr);

at::Tensor linear_inplace_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const ideep::attr_t& attr);

// IPEX customized linear OP with n-D packed weight
// Additional out_features, in_features is used to query expected weigth desc
// Since n-D packed weight have loss these info
class IPEXLinearOp : public torch::autograd::Function<IPEXLinearOp> {
public:
  // forward function without autograd overhead, will go this way when only do forward
  static at::Tensor _forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const at::Tensor& bias = at::Tensor());

  static at::Tensor forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const at::Tensor& bias = at::Tensor());

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward_impl(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    std::array<bool,3> output_mask);

at::Tensor linear_forward_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const at::Tensor& bias,
    const ideep::attr_t& attr);

at::Tensor ipex_linear(
    const at::Tensor& self,
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const c10::optional<at::Tensor>& bias_opt);

}  // namespace cpu
}  // namespace torch_ipex
