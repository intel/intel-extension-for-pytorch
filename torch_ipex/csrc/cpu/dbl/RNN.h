#pragma once

#include <ATen/Tensor.h>

#include "cpu/dil/dil.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace rnn {

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>> mkldnn_rnn(
    const at::Tensor& input_, std::vector<at::Tensor> weight, int64_t weight_stride0,
    const at::Tensor& hx_, const at::Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, at::IntArrayRef batch_sizes);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mkldnn_rnn_backward(
    const at::Tensor& input_, std::vector<at::Tensor> weight, int64_t weight_stride0,
    const at::Tensor& hx_, const at::Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, at::IntArrayRef batch_sizes,
    std::vector<at::Tensor> outputs, const at::Tensor& grad_output, const at::Tensor& grad_hy, const at::Tensor& grad_cy, std::vector<at::Tensor> layer_outputs);
}  // namespace rnn
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
