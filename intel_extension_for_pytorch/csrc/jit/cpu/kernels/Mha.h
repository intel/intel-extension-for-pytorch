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

at::Tensor dil_distil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const int64_t& transpose_dim_a,
    const int64_t& transpose_dim_b,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head);

at::Tensor dil_maskedfill_softmax(
    at::Tensor& qk,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill);
} // namespace cpu
} // namespace torch_ipex
