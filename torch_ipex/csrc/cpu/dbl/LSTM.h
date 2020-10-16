#pragma once

#include <ATen/Tensor.h>

#include "cpu/dil/dil.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace lstm {

std::pair<at::Tensor, std::tuple<at::Tensor, at::Tensor>> lstm_impl(
    const at::Tensor& input, std::vector<at::Tensor>  hidden,
    std::vector<at::Tensor> params, bool has_biases, dil::rnn_kind mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_backward_impl(
    const at::Tensor& input, std::vector<at::Tensor> hidden,
    std::vector<at::Tensor> params, bool has_biases, dil::rnn_kind mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first,
    std::vector<at::Tensor> outputs, const at::Tensor& grad_output, const at::Tensor& grad_hy, const at::Tensor& grad_cy);
}  // namespace lstm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
