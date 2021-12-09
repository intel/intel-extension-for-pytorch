#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor dil_qinteraction(
    const std::vector<at::Tensor> input,
    double o_scale,
    int64_t o_zp,
    at::ScalarType o_dtype);

} // namespace cpu
} // namespace torch_ipex
