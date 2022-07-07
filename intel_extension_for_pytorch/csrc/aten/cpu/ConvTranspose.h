#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/jit/cpu/kernels/OpContext.h"

namespace torch_ipex {
namespace cpu {

at::Tensor conv_transpose_kernel_impl(
    const at::Tensor& input,
    const ideep::tensor& w,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef origin_weight_dims,
    const ideep::attr_t& attr);

void conv_transpose_out_kernel_impl(
    const at::Tensor& input,
    const ideep::tensor& w,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef origin_weight_dims,
    const ideep::attr_t& attr);

at::Tensor conv_transpose(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
conv_transpose_backward_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& at_weight,
    const ideep::tensor& packed_weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    std::array<bool, 3> output_mask,
    bool weight_channels_last);

class IPEXConvTransposeOp
    : public torch::autograd::Function<IPEXConvTransposeOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      const at::Tensor& op_context);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      const at::Tensor& op_context);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

} // namespace cpu
} // namespace torch_ipex
