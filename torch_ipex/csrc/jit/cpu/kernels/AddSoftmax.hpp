#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace kernels {

// This operator assumes that the softmax is applied to the last
// dimension and the alpha value for the add is 1.0f. Besides that,
// the operator only support FP32 now.
at::Tensor AddSoftmax(const at::Tensor& a, const at::Tensor& b);

} // namespace kernels
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
