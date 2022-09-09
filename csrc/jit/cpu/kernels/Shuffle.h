#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor dil_shuffle(
    const at::Tensor& self,
    at::IntArrayRef view_shape,
    int64_t dim0,
    int64_t dim1);

} // namespace cpu
} // namespace torch_ipex
