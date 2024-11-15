#ifdef USE_XETLA_XE_HPC

#pragma once
#include "../../xetla_kernel_api.h"

namespace torch_ipex::xpu::xetla::xehpc {

int hgemm_find_policy_id(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);
int hgemm_qkv_find_policy_id(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

cgfs_t hgemm_addmm(
    const int policy_id,
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
    const float beta);

cgfs_t hgemm_common(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k);

cgfs_t hgemm_res(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res_factor);

cgfs_t hgemm_res_res(
    const int policy_id,
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
    const float res1_factor);

cgfs_t hgemm_bias(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor);

cgfs_t hgemm_bias_res(
    const int policy_id,
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
    const float res_factor);

cgfs_t hgemm_bias_res_res(
    const int policy_id,
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
    const float res1_factor);

cgfs_t hgemm_bias_relu(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor);

cgfs_t hgemm_bias_gelu(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor);

cgfs_t hgemm_resmul(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k);

cgfs_t hgemm_silu(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k);

cgfs_t hgemm_qkv(
    const int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k);

cgfs_t hgemm_qkv_bias(
    const int policy_id,
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
    const int k);

cgfs_t hgemm_qkv_group(
    const int policy_id,
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
    const int head_dim);

cgfs_t hgemm_qkv_group_bias(
    const int policy_id,
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
    const int head_dim);

} // namespace torch_ipex::xpu::xetla::xehpc

#endif
