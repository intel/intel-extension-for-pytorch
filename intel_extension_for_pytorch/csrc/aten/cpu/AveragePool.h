#pragma once

#include <ATen/ATen.h>

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
void cpu_avg_pool(
    const at::Tensor& output_,
    const at::Tensor& input_,
    int64_t kW,
    int64_t kH,
    int64_t dW,
    int64_t dH,
    int64_t padW,
    int64_t padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

template <typename scalar_t>
void cpu_avg_pool_channels_last(
    const at::Tensor& output_,
    const at::Tensor& input_,
    int64_t kW,
    int64_t kH,
    int64_t dW,
    int64_t dH,
    int64_t padW,
    int64_t padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

template <>
void cpu_avg_pool_channels_last<at::BFloat16>(
    const at::Tensor& output_,
    const at::Tensor& input_,
    int64_t kW,
    int64_t kH,
    int64_t dW,
    int64_t dH,
    int64_t padW,
    int64_t padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

template <typename scalar_t>
void cpu_avg_pool_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

template <typename scalar_t>
void cpu_avg_pool_backward_channels_last(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

void avg_pool2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    int64_t kW,
    int64_t kH,
    int64_t dW,
    int64_t dH,
    int64_t padW,
    int64_t padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

void avg_pool2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

at::Tensor avg_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

at::Tensor avg_pool2d_backward_out_cpu(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

} // namespace cpu
} // namespace torch_ipex
