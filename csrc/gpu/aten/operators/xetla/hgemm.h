#pragma once
#include <sycl/sycl.hpp>
#include "xetla_kernel_api.h"

namespace xpu::xetla {

enum class GemmStatus { kSuccess, kError };

XETLA_KERNEL_API cgfs_t hgemm_addmm(
    int policy_id,
    sycl::half* out,
    const sycl::half* res,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_common(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_res(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_res_res(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res0,
    const sycl::half* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res0_factor,
    const float res1_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_bias(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_bias_res(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_bias_res_res(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res0,
    const sycl::half* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res0_factor,
    const float res1_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_bias_relu(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_bias_gelu(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_resmul(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_silu(
    int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_qkv(
    int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_qkv_bias(
    int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_qkv_group(
    int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim,
    const bool is_b_row_major);

XETLA_KERNEL_API cgfs_t hgemm_qkv_group_bias(
    int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim,
    const bool is_b_row_major);

} // namespace xpu::xetla
