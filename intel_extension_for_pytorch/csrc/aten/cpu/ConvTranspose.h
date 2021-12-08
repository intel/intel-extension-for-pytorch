#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor conv_transpose2d_kernel_impl(
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

at::Tensor convolution_transpose_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size = {},
    int64_t output_channel = -1,
    bool weight_channels_last = false,
    bool weight_prepacked = false);

at::Tensor conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t output_channel,
    bool weight_channels_last,
    bool weight_prepacked);

class IPEXConvTransposeOp
    : public torch::autograd::Function<IPEXConvTransposeOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef output_padding,
      int64_t groups,
      at::IntArrayRef dilation,
      at::IntArrayRef kernel_size,
      int64_t output_channel,
      bool weight_channels_last,
      bool weight_prepacked);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef output_padding,
      int64_t groups,
      at::IntArrayRef dilation,
      at::IntArrayRef kernel_size,
      int64_t output_channel,
      bool weight_channels_last,
      bool weight_prepacked);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

} // namespace cpu
} // namespace torch_ipex
