#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include <ideep.hpp>

namespace torch_ipex {
namespace cpu {

// int8 op
at::Tensor dil_qembeddingbag(
    const at::Tensor weight,
    const at::Tensor indices,
    const at::Tensor offsets,
    bool sparse,
    bool include_last_offset,
    double o_scale,
    int64_t o_zp,
    at::ScalarType o_dtype);

} // namespace cpu
} // namespace torch_ipex
