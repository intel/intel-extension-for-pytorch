#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> dil_native_layer_norm_impl(
    const at::Tensor &X, const at::Tensor &gamma /* optional */,
    const at::Tensor &beta /* optional */, int64_t M, int64_t N, double eps);
} // namespace cpu
} // namespace torch_ipex
