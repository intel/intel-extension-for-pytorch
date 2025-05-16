#pragma once

#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/csrc/autograd/custom_function.h>

#include <ideep.hpp>
#include "cpu/kernels/OpContext.h"

namespace torch_ipex {
namespace cpu {

void convolution_kernel_output(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& bias_opt,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr);

at::Tensor convolution_kernel(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr,
    at::MemoryFormat memory_format);

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_kernel(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& at_weight,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& mkldnn_bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const bool weight_channels_last,
    std::array<bool, 3> output_mask);

std::vector<int64_t> calc_conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation);

std::tuple<at::Tensor, at::Tensor> causal_conv1d_update(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens);

std::tuple<at::Tensor, at::Tensor> causal_conv1d_fn(
    const at::Tensor& x,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    const c10::optional<at::Tensor>& initial_states,
    const c10::optional<at::Tensor>& final_states_out,
    bool silu_activation);

// IPEX customized convolution OP with n-D packed weight
class IPEXConvolutionOp : public torch::autograd::Function<IPEXConvolutionOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      const at::Tensor& op_context,
      c10::optional<at::IntArrayRef> kernel_size,
      c10::optional<at::IntArrayRef> padding,
      c10::optional<at::IntArrayRef> stride,
      c10::optional<at::IntArrayRef> dilation,
      c10::optional<bool> weight_channels_last);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt,
      const at::Tensor& op_context,
      c10::optional<at::IntArrayRef> kernel_size,
      c10::optional<at::IntArrayRef> padding,
      c10::optional<at::IntArrayRef> stride,
      c10::optional<at::IntArrayRef> dilation,
      c10::optional<bool> weight_channels_last);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context,
    c10::optional<at::IntArrayRef> kernel_size,
    c10::optional<at::IntArrayRef> padding,
    c10::optional<at::IntArrayRef> stride,
    c10::optional<at::IntArrayRef> dilation,
    c10::optional<bool> weight_channels_last);

using causal_conv1d_update_kernel_fn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens);
using causal_conv1d_fn_kernel_fn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor& x,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    const c10::optional<at::Tensor>& initial_states,
    const c10::optional<at::Tensor>& final_states_out,
    bool silu_activation);
IPEX_DECLARE_DISPATCH(
    causal_conv1d_update_kernel_fn,
    causal_conv1d_update_kernel_stub);
IPEX_DECLARE_DISPATCH(causal_conv1d_fn_kernel_fn, causal_conv1d_fn_kernel_stub);
} // namespace cpu
} // namespace torch_ipex
