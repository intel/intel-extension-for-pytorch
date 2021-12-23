#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

// Get conv packed weight according to input_size,
// if input size is empty, will use dummy input size, the
// weight_is_channels_last works when weight_packed=true, and
// use_channels_last only works when input_size is none-empty, it will force
// weight to channels last when use_channels_last is true given a input size.
ideep::tensor get_conv_packed_weight(
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef weight_size,
    int64_t groups,
    bool weight_is_channels_last,
    bool weight_packed,
    bool use_channels_last,
    at::IntArrayRef input_size,
    const ideep::attr_t& attr);

// Get the linear's expected ideep weight tensor, the weight may be a 2-D tensor
// or has benn packed to a n-D tensor, if it is a plain tensor, it will reorder
// to a expected weight according queried desc of OneDNN linear, or if it is
// pack, it will init a ideep tensor according queried desc and weight's
// data_ptr(not has memory copy).
ideep::tensor get_linear_packed_weight(
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features);

std::tuple<ideep::tensor, ideep::tensor> get_lstm_packed_weight(
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
    const bool reverse,
    const bool train);

bool is_packed(const at::Tensor& weight);

// pack linear's weight according to dummy input.
// weight: weight need to be packed
// dtype: dtype used to query best weight format

at::Tensor linear_weight_pack(
    const at::Tensor& weight,
    c10::optional<at::ScalarType> dtype);

// Unpack Linear's weight according to dummy input
at::Tensor linear_weight_unpack(
    const at::Tensor& weight,
    const int64_t out_features,
    const int64_t in_features,
    const bool original_weight_transposed,
    c10::optional<at::ScalarType> dtype);

ideep::tensor get_conv_transpose2d_packed_weight(
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef weight_size,
    int64_t groups,
    bool weight_is_channels_last,
    bool weight_packed,
    bool use_channels_last,
    at::IntArrayRef input_size,
    const ideep::attr_t& attr);

at::Tensor conv_transpose2d_weight_pack(
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation,
    c10::optional<at::ScalarType> dtype);

} // namespace cpu
} // namespace torch_ipex
