#pragma once

#include <ATen/Tensor.h>

#include <vector>

namespace torch_ipex {
namespace cpu {

at::Tensor mean_dim_impl(
    const at::Tensor& input,
    c10::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);

} // namespace cpu
} // namespace torch_ipex
