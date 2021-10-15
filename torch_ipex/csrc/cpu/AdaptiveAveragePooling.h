#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu {

static inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::floor((float)(a * c) / b);
}

static inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::ceil((float)((a + 1) * c) / b);
}

template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_avg_pool(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef output_size);

template <typename scalar_t>
void cpu_adaptive_avg_pool_channels_last(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef output_size);

template <>
void cpu_adaptive_avg_pool_channels_last<at::BFloat16>(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef output_size);

template <typename scalar_t>
void cpu_adaptive_avg_pool_backward(
    at::Tensor& grad_input_,
    const at::Tensor& grad_output_);

template <typename scalar_t>
void cpu_adaptive_avg_pool_backward_channels_last(
    at::Tensor& grad_input_,
    const at::Tensor& grad_output_);

void adaptive_avg_pool2d_kernel_impl(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef output_size);

void adaptive_avg_pool2d_backward_kernel_impl(
    at::Tensor& grad_input,
    const at::Tensor& grad_output);

void adaptive_avg_pool2d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    at::IntArrayRef output_size);

at::Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input);

at::Tensor& adaptive_avg_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::Tensor& output);

at::Tensor adaptive_avg_pool2d_cpu(
    at::Tensor const& input,
    at::IntArrayRef output_size);

at::Tensor adaptive_avg_pool2d(
    at::Tensor const& input,
    at::IntArrayRef output_size);

at::Tensor& adaptive_avg_pool2d_backward_out_cpu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input);

at::Tensor adaptive_avg_pool2d_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input);

} // namespace cpu
} // namespace torch_ipex