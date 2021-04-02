#pragma once

#include <ATen/Tensor.h>

#include <ideep.hpp>

#include <vector>

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
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr);

void convolution_inplace_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr);

// void prepack_conv_weights(
//     const at::Tensor& input,
//     const at::Tensor& dil_input,
//     const at::Tensor& weight,
//     at::IntArrayRef stride,
//     at::IntArrayRef padding,
//     at::IntArrayRef dilation,
//     int64_t groups);

}  // namespace cpu
}  // namespace torch_ipex
