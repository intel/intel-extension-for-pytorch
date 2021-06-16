#pragma once

#include <ATen/ATen.h>

#include "cpu/dil/dil.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace deconv {

std::vector<int64_t> calc_deconv_input_size(
    at::IntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups);

std::vector<int64_t> calc_padding_r_adjusted(
    const int64_t input_dims,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding);

dil::tensor deconvolution_impl(
    const dil::tensor& x,
    const dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    at::IntArrayRef padding,
    std::vector<int64_t>  padding_r,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& attr = dil::attr_t());

void prepack_deconv_weights(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    std::vector<int64_t>  padding_r,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    int64_t groups,
    bool with_bias);

}  // namespace deconv
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
