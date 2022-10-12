#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include <ideep.hpp>

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
    const at::Scalar& fill,
    const at::Scalar& dim_per_head);

at::Tensor dil_vit_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& mask_qk_reshp,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head);

at::Tensor dil_maskedfill_softmax(
    at::Tensor& qk,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill);

at::Tensor dil_transfree_mha(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& head_num,
    const int64_t& head_size);

at::Tensor dil_transfree_distil_mha(
    const at::Tensor& qkv,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head,
    const int64_t& head_num,
    const int64_t& head_size);

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const at::Tensor& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& head_num,
    const int64_t& head_size);

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const double& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& head_num,
    const int64_t& head_size);

at::Tensor dil_mha_matmul_trans(
    const at::Tensor& left,
    const at::Tensor& right);

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> dil_qkv_split(
    const at::Tensor& qkv);

} // namespace cpu
} // namespace torch_ipex
