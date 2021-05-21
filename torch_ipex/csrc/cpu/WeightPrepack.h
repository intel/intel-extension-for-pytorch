#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

// Get the convolution's expected ideep weight tensor, may be a block weight.
// if the expected weight doesn't not exist, it will create an expected weight according
// to the queried desc of OneDNN conv, and the expected weight will be cached.
// TODO: if weight is a block weight, direct return the weight(ideep tensor).

// input: an ideep tensor, getting from the convolution's input,
// weight: convolution's weight
// stride, padding, dilation, groups: convolution's attribute.
// attr: for fuse op.
ideep::tensor get_conv_prepacked_weight(
    const ideep::tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr);

// Get the linear's expected ideep weight tensor, may be a block weight.
// if the expected weight doesn't not exist, it will create an expected weight according
// to the queried desc of OneDNN linear, and the expected weight will be cached.
// TODO: if weight is a block weight, direct return the weight(ideep tensor).

// input: an ideep tensor, getting from the linear's input,
// weight: linear's weight
ideep::tensor get_linear_prepacked_weight(
    const ideep::tensor& input,
    const at::Tensor& weight);

} // namespace cpu
}  // namespace torch_ipex
