#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace kernels {

// This operator assumes that the softmax is applied to the last
// dimension.
at::Tensor DivAddSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head);

} // namespace kernels
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
