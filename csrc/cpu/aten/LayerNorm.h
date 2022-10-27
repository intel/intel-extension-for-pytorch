#pragma once

#include <ATen/Tensor.h>

#include <ideep.hpp>

namespace torch_ipex {
namespace cpu {

at::Tensor layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps,
    bool cudnn_enable);

} // namespace cpu
} // namespace torch_ipex
