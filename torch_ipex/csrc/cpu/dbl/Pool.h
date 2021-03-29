#pragma once

#include <ATen/ATen.h>

#include "cpu/dil/dil.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace pool {

template<typename T>
static inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode);

std::vector<int64_t> pool_output_sizes(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding_l,
    at::IntArrayRef padding_r,
    at::IntArrayRef dilation,
    bool ceil_mode);

at::Tensor _dil_pooling(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    dil::algorithm algo);

at::Tensor _dil_pooling_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    dil::algorithm algo);

}  // namespace pool
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
