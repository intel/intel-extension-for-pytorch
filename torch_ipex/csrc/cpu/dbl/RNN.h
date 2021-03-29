#pragma once

#include <ATen/ATen.h>

#include "cpu/dil/dil.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace rnn {

std::vector<at::Tensor> mkldnn_rnn_layer(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& w3, const at::Tensor& w4, const at::Tensor& hx_, const at::Tensor& cx_, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes);

std::vector<at::Tensor> mkldnn_rnn_layer_backward(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& w3, const at::Tensor& w4, const at::Tensor& hx_, const at::Tensor& cx_, const at::Tensor& output, const at::Tensor& hy_,
    const at::Tensor& cy_, const at::Tensor& grad_output, const at::Tensor& grad_hy_, const at::Tensor& grad_cy_, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes);
}  // namespace rnn
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
