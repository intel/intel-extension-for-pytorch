#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

template <typename dest_t, typename src_t>
static inline dest_t safe_downcast(src_t v) {
  TORCH_CHECK(
      std::numeric_limits<dest_t>::min() <= v &&
          v <= std::numeric_limits<dest_t>::max(),
      "integer out of range");

  return static_cast<dest_t>(v);
}

template <typename scalar_t, typename accscalar_t>
void cpu_max_pool(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

template <typename scalar_t>
void cpu_max_pool_channels_last(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

template <>
void cpu_max_pool_channels_last<at::BFloat16>(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

template <typename scalar_t>
void cpu_max_pool_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const at::Tensor& indices_);

template <typename scalar_t>
void cpu_max_pool_backward_channels_last(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const at::Tensor& indices_);

void max_pool2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& indices,
    const at::Tensor& input,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

void max_pool2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& indices);

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);

at::Tensor max_pool2d_with_indices_backward_out_cpu(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices);

} // namespace cpu
} // namespace torch_ipex