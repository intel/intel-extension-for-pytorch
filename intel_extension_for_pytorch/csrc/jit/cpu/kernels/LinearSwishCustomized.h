#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor dil_linear_swish_customized(
    at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias);
} // namespace cpu
} // namespace torch_ipex
