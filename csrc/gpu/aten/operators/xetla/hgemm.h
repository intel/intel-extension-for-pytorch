#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

enum class GemmStatus { kSuccess, kError };

GemmStatus hgemm_addmm(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* res,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta,
    const bool is_b_row_major);

GemmStatus hgemm_common(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

GemmStatus hgemm_res(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res,
    const int m,
    const int n,
    const int k,
    const float res_factor,
    const bool is_b_row_major);

GemmStatus hgemm_res_res(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res0,
    const sycl::half* res1,
    const int m,
    const int n,
    const int k,
    const float res0_factor,
    const float res1_factor,
    const bool is_b_row_major);

GemmStatus hgemm_bias(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const bool is_b_row_major);

GemmStatus hgemm_bias_res(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res_factor,
    const bool is_b_row_major);

GemmStatus hgemm_bias_res_res(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res0,
    const sycl::half* res1,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res0_factor,
    const float res1_factor,
    const bool is_b_row_major);

GemmStatus hgemm_bias_relu(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const bool is_b_row_major);

GemmStatus hgemm_bias_gelu(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const bool is_b_row_major);

GemmStatus hgemm_resmul(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* mul,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

GemmStatus hgemm_silu(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

GemmStatus hgemm_qkv(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

GemmStatus hgemm_qkv_bias(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major);

GemmStatus hgemm_qkv_group(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim,
    const bool is_b_row_major);

GemmStatus hgemm_qkv_group_bias(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim,
    const bool is_b_row_major);

} // namespace xetla
} // namespace xpu
