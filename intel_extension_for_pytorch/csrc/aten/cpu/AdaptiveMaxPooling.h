#pragma once

#include <ATen/ATen.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

static inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::floor((float)(a * c) / b);
}

static inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::ceil((float)((a + 1) * c) / b);
}

template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_max_pool(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    at::IntArrayRef output_size);

template <typename scalar_t>
void cpu_adaptive_max_pool_channels_last(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    at::IntArrayRef output_size);

template <>
void cpu_adaptive_max_pool_channels_last<at::BFloat16>(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    at::IntArrayRef output_size);

template <typename scalar_t>
void cpu_adaptive_max_pool_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const at::Tensor& indices_);

template <typename scalar_t>
void cpu_adaptive_max_pool_backward_channels_last(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const at::Tensor& indices_);

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size);

at::Tensor adaptive_max_pool2d_backward_out_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& indices);

namespace {

void adaptive_max_pool2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& indices,
    const at::Tensor& input,
    at::IntArrayRef output_size);

void adaptive_max_pool2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& indices);
} // namespace

using adaptive_max_pool2d_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::IntArrayRef);
DECLARE_DISPATCH(
    adaptive_max_pool2d_kernel_fn,
    adaptive_max_pool2d_kernel_stub);

using adaptive_max_pool2d_backward_kernel_fn =
    void (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);
DECLARE_DISPATCH(
    adaptive_max_pool2d_backward_kernel_fn,
    adaptive_max_pool2d_backward_kernel_stub);

} // namespace cpu
} // namespace torch_ipex