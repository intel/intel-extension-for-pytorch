#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

// Get the convolution's expected ideep weight tensor, may be a block weight.
// if the expected weight doesn't exist, it will create an expected weight according
// to the queried desc of OneDNN conv, and the expected weight will be cached.
// TODO: if weight is already in the expected format, return the weight(ideep tensor) directly.

// input: an ideep tensor, getting from the convolution's input,
// weight: convolution's weight
// stride, padding, dilation, groups: convolution's attribute.
// attr: for fuse op.
ideep::tensor get_conv_prepacked_weight(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr,
    const at::MemoryFormat& mkldnn_memory_format);

// Get the convolution's expected ideep weight tensor, the weight may be a plain tensor
// or has benn prepacked to a n-D tensor, if it is a plain tensor, it will reorder to a
// expected weight according queried desc of OneDNN conv, or if it is prepack, it will init
// a ideep tensor according queried desc and weight's data_ptr(not has memory copy).
ideep::tensor get_conv_prepacked_weight(
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    int64_t input_channel,
    bool is_channels_last,
    bool weight_prepacked);

// Prepack convolution's weight according to dummy input.
// weight: weight need to be prepacked
// dtype: if given dtype, will use this dtype to override weight's dtype

at::Tensor conv2d_weight_prepack(
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    c10::optional<at::ScalarType> dtype);

// Unpack convolution's weight according to dummy input.
at::Tensor conv2d_weight_unpack(
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    at::IntArrayRef kernel_size,
    int64_t groups,
    int64_t output_channel,
    int64_t input_channel,
    bool is_channels_last,
    c10::optional<at::ScalarType> dtype);

// Get the linear's expected ideep weight tensor, may be a block weight.
// if the expected weight doesn't exist, it will create an expected weight according
// to the queried desc of OneDNN linear, and the expected weight will be cached.
// TODO: if weight is already in the expected format, return the weight(ideep tensor) directly.

// input: an ideep tensor, getting from the linear's input,
// weight: linear's weight
ideep::tensor get_linear_prepacked_weight(
    const at::Tensor& weight,
    const int64_t batch_size,
    const at::ScalarType src_dtype);

// Get the linear's expected ideep weight tensor, the weight may be a 2-D tensor
// or has benn prepacked to a n-D tensor, if it is a plain tensor, it will reorder to a
// expected weight according queried desc of OneDNN linear, or if it is prepack, it will init
// a ideep tensor according queried desc and weight's data_ptr(not has memory copy).
ideep::tensor get_linear_prepacked_weight(
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features);

std::tuple<ideep::tensor, ideep::tensor> get_lstm_prepacked_weight(
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    int64_t input_size,
    int64_t num_gates,
    int64_t hidden_size,
    const ideep::dims& output_sizes,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& bias,
    const bool reverse);

bool is_prepacked(const at::Tensor& weight);

// Prepack linear's weight according to dummy input.
// weight: weight need to be prepacked
// dtype: dtype used to query best weight format

at::Tensor linear_weight_prepack(
    const at::Tensor& weight,
    c10::optional<at::ScalarType> dtype);

// Unpack Linear's weight according to dummy input
at::Tensor linear_weight_unpack(
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const bool original_weight_transposed,
    c10::optional<at::ScalarType> dtype);

} // namespace cpu
}  // namespace torch_ipex
