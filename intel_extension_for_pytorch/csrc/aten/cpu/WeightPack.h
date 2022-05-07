#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

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

// Get the convolution's expected ideep weight tensor desc.
ideep::tensor::desc get_conv_expected_weights_desc(
    const ideep::tensor::dims& weights_dims,
    ideep::tensor::data_type w_dtype = ideep::data_type::f32,
    const ideep::tensor::dims& strides = {1, 1, 1},
    const ideep::tensor::dims& padding_l = {0, 0, 0},
    const ideep::tensor::dims& padding_r = {0, 0, 0},
    const ideep::tensor::dims& dilates = {1, 1, 1},
    int groups = 1,
    bool channels_last = false,
    ideep::algorithm aalgorithm = ideep::algorithm::convolution_direct,
    ideep::data_type x_dtype = ideep::data_type::f32,
    const ideep::dims& src_dims = ideep::tensor::dims(),
    const ideep::attr_t& attr = ideep::attr_t());

// Get the conv_transpose's expected ideep weight tensor desc.
ideep::tensor::desc get_conv_transpose_expected_weights_desc(
    const ideep::tensor::dims& weights_dims,
    ideep::tensor::data_type w_dtype = ideep::data_type::f32,
    const ideep::tensor::dims& strides = {1, 1, 1},
    const ideep::tensor::dims& padding_l = {0, 0, 0},
    const ideep::tensor::dims& padding_r = {0, 0, 0},
    const ideep::tensor::dims& dilates = {1, 1, 1},
    int groups = 1,
    bool channels_last = false,
    ideep::algorithm aalgorithm = ideep::algorithm::deconvolution_direct,
    ideep::data_type x_dtype = ideep::data_type::f32,
    const ideep::dims& src_dims = ideep::tensor::dims(),
    const ideep::attr_t& attr = ideep::attr_t());

} // namespace cpu
} // namespace torch_ipex
