#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor dil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype);

} // namespace cpu
} // namespace torch_ipex
