#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

std::vector<int64_t> calc_conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation);

at::Tensor convolution_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr);

void convolution_inplace_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr);

// IPEX customized linear OP with n-D packed weight
class IPEXConvolutionOp : public torch::autograd::Function<IPEXConvolutionOp> {
public:
  // forward function without autograd overhead, will go this way when only do forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      at::IntArrayRef kernel_size,
      int64_t groups,
      int64_t output_channel,
      bool weight_channels_last);

 static at::Tensor forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_channels_last);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list  grad_outputs);

};


}  // namespace cpu
}  // namespace torch_ipex
