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

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const at::Tensor& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& num_head,
    const int64_t& headSize);

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const double& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& num_head,
    const int64_t& headSize);

at::Tensor dil_mha_matmul_trans(
    const at::Tensor& left,
    const at::Tensor& right);

at::Tensor dil_bert_flash_mha(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& num_head,
    const int64_t& headSize);

/**
 * For one kind of SD MHA, the query/key/value linears are fused by
 * the ConcatLinear. Here the "split_list" stores the sizes of the
 * dims which are connected. It is used to calculate MHA's head size.
 */
at::Tensor dil_sd_flash_mha(
    const at::Tensor& qkv,
    const at::IntArrayRef& split_list,
    const at::IValue& scale,
    const int64_t& num_head);

at::Tensor dil_sd_flash_mha(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::IValue& scale,
    const int64_t& num_head);

template <typename T>
std::vector<at::Tensor> dil_mat_split(
    const at::Tensor& qkv,
    const at::IntArrayRef& split_list);

c10::List<at::Tensor> dil_split_tensor(
    const at::Tensor& mat,
    const at::IntArrayRef& split_list);

} // namespace cpu
} // namespace torch_ipex
