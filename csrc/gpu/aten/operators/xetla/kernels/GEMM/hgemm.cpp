#ifdef USE_XETLA_XE_HPC
#include "../../hgemm.h"
#include <iostream>
#include "hgemm_impl.h"
#include "hgemm_policy.h"
namespace torch_ipex::xpu::xetla {

// clang-format off

#define HGEMM_IMPL_NAME(HEAD, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HEAD##_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

#define HGEMM_ADDMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_addmm, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_COMMON_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_common, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_BIAS_RELU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_relu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_GELU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_gelu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RESMUL_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_resmul, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_SILU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_silu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_QKV_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_GROUP_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv_group, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_GROUP_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv_group_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

// clang-format on

#define HGEMM_ENUMERATE_IMPLS(                                                 \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)                         \
  cgfs_t HGEMM_ADDMM_IMPL_NAME(                                                \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* res,                                                   \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float alpha,                                                       \
      const float beta) {                                                      \
    return hgemm_addmm<                                                        \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, res, a, b, acc_ptr, cnt_ptr, m, n, k, alpha, beta);  \
  }                                                                            \
  cgfs_t HGEMM_COMMON_IMPL_NAME(                                               \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    return hgemm_common<                                                       \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, acc_ptr, cnt_ptr, m, n, k);                    \
  }                                                                            \
  cgfs_t HGEMM_RES_IMPL_NAME(                                                  \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* res,                                                   \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float res_factor) {                                                \
    return hgemm_res<                                                          \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, res, acc_ptr, cnt_ptr, m, n, k, res_factor);   \
  }                                                                            \
  cgfs_t HGEMM_RES_RES_IMPL_NAME(                                              \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* res0,                                                  \
      const sycl::half* res1,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float res0_factor,                                                 \
      const float res1_factor) {                                               \
    return hgemm_res_res<                                                      \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(                                                          \
        out,                                                                   \
        a,                                                                     \
        b,                                                                     \
        res0,                                                                  \
        res1,                                                                  \
        acc_ptr,                                                               \
        cnt_ptr,                                                               \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        res0_factor,                                                           \
        res1_factor);                                                          \
  }                                                                            \
  cgfs_t HGEMM_BIAS_IMPL_NAME(                                                 \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor) {                                               \
    return hgemm_bias<                                                         \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor); \
  }                                                                            \
  cgfs_t HGEMM_BIAS_RES_IMPL_NAME(                                             \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res,                                                   \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor,                                                 \
      const float res_factor) {                                                \
    return hgemm_bias_res<                                                     \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(                                                          \
        out,                                                                   \
        a,                                                                     \
        b,                                                                     \
        bias,                                                                  \
        res,                                                                   \
        acc_ptr,                                                               \
        cnt_ptr,                                                               \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        bias_factor,                                                           \
        res_factor);                                                           \
  }                                                                            \
  cgfs_t HGEMM_BIAS_RES_RES_IMPL_NAME(                                         \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res0,                                                  \
      const sycl::half* res1,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor,                                                 \
      const float res0_factor,                                                 \
      const float res1_factor) {                                               \
    return hgemm_bias_res_res<                                                 \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(                                                          \
        out,                                                                   \
        a,                                                                     \
        b,                                                                     \
        bias,                                                                  \
        res0,                                                                  \
        res1,                                                                  \
        acc_ptr,                                                               \
        cnt_ptr,                                                               \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        bias_factor,                                                           \
        res0_factor,                                                           \
        res1_factor);                                                          \
  }                                                                            \
  cgfs_t HGEMM_BIAS_RELU_IMPL_NAME(                                            \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor) {                                               \
    return hgemm_bias_relu<                                                    \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor); \
  }                                                                            \
  cgfs_t HGEMM_BIAS_GELU_IMPL_NAME(                                            \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor) {                                               \
    return hgemm_bias_gelu<                                                    \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor); \
  }                                                                            \
  cgfs_t HGEMM_RESMUL_IMPL_NAME(                                               \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* mul,                                                   \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    return hgemm_mul<                                                          \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, mul, acc_ptr, cnt_ptr, m, n, k);               \
  }                                                                            \
  cgfs_t HGEMM_SILU_IMPL_NAME(                                                 \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    return hgemm_silu<                                                         \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out, a, b, acc_ptr, cnt_ptr, m, n, k);                    \
  }                                                                            \
  cgfs_t HGEMM_QKV_IMPL_NAME(                                                  \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    return hgemm_qkv<                                                          \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out0, out1, out2, a, b, acc_ptr, cnt_ptr, m, n, k);       \
  }                                                                            \
  cgfs_t HGEMM_QKV_BIAS_IMPL_NAME(                                             \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    return hgemm_qkv_bias<                                                     \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(out0, out1, out2, a, b, bias, acc_ptr, cnt_ptr, m, n, k); \
  }                                                                            \
  cgfs_t HGEMM_QKV_GROUP_IMPL_NAME(                                            \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const int num_kv_head,                                                   \
      const int group,                                                         \
      const int head_dim) {                                                    \
    return hgemm_qkv_group<                                                    \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(                                                          \
        out0,                                                                  \
        out1,                                                                  \
        out2,                                                                  \
        a,                                                                     \
        b,                                                                     \
        acc_ptr,                                                               \
        cnt_ptr,                                                               \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        num_kv_head,                                                           \
        group,                                                                 \
        head_dim);                                                             \
  }                                                                            \
  cgfs_t HGEMM_QKV_GROUP_BIAS_IMPL_NAME(                                       \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      float* acc_ptr,                                                          \
      uint32_t* cnt_ptr,                                                       \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const int num_kv_head,                                                   \
      const int group,                                                         \
      const int head_dim) {                                                    \
    return hgemm_qkv_group_bias<                                               \
        sycl::half,                                                            \
        WG_M,                                                                  \
        WG_N,                                                                  \
        SG_M,                                                                  \
        SG_N,                                                                  \
        SG_K,                                                                  \
        SLM_KS,                                                                \
        1,                                                                     \
        1,                                                                     \
        3,                                                                     \
        B_ROW_MAJOR>(                                                          \
        out0,                                                                  \
        out1,                                                                  \
        out2,                                                                  \
        a,                                                                     \
        b,                                                                     \
        bias,                                                                  \
        acc_ptr,                                                               \
        cnt_ptr,                                                               \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        num_kv_head,                                                           \
        group,                                                                 \
        head_dim);                                                             \
  }

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_IMPLS)

cgfs_t (*hgemm_addmm_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_ADDMM_IMPL_NAME)};

cgfs_t (*hgemm_common_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_COMMON_IMPL_NAME)};

cgfs_t (*hgemm_res_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_IMPL_NAME)};

cgfs_t (*hgemm_res_res_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_RES_IMPL_NAME)};

cgfs_t (*hgemm_bias_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_IMPL_NAME)};

cgfs_t (*hgemm_bias_res_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_IMPL_NAME)};

cgfs_t (*hgemm_bias_res_res_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float,
    const float,
    const float) = {
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_RES_IMPL_NAME)};

cgfs_t (*hgemm_bias_relu_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RELU_IMPL_NAME)};

cgfs_t (*hgemm_bias_gelu_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_GELU_IMPL_NAME)};

cgfs_t (*hgemm_resmul_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RESMUL_IMPL_NAME)};

cgfs_t (*hgemm_silu_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_SILU_IMPL_NAME)};

cgfs_t (*hgemm_qkv_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_IMPL_NAME)};

cgfs_t (*hgemm_qkv_bias_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_BIAS_IMPL_NAME)};

cgfs_t (*hgemm_qkv_group_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_GROUP_IMPL_NAME)};

cgfs_t (*hgemm_qkv_group_bias_policies[HGEMM_NUM_POLICIES])(
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    float*,
    uint32_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int) = {
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_GROUP_BIAS_IMPL_NAME)};

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_addmm_policies[policy_id](
      out, res, a, b, acc_ptr, cnt_ptr, m, n, k, alpha, beta);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_common_policies[policy_id](out, a, b, acc_ptr, cnt_ptr, m, n, k);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_res_policies[policy_id](
      out, a, b, res, acc_ptr, cnt_ptr, m, n, k, res_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_res_res_policies[policy_id](
      out,
      a,
      b,
      res0,
      res1,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      res0_factor,
      res1_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_bias_policies[policy_id](
      out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_bias_res_policies[policy_id](
      out, a, b, bias, res, acc_ptr, cnt_ptr, m, n, k, bias_factor, res_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_bias_res_res_policies[policy_id](
      out,
      a,
      b,
      bias,
      res0,
      res1,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      bias_factor,
      res0_factor,
      res1_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_bias_relu_policies[policy_id](
      out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_bias_gelu_policies[policy_id](
      out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_resmul_policies[policy_id](
      out, a, b, mul, acc_ptr, cnt_ptr, m, n, k);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_silu_policies[policy_id](out, a, b, acc_ptr, cnt_ptr, m, n, k);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_qkv_policies[policy_id](
      out0, out1, out2, a, b, acc_ptr, cnt_ptr, m, n, k);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_qkv_bias_policies[policy_id](
      out0, out1, out2, a, b, bias, acc_ptr, cnt_ptr, m, n, k);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_qkv_group_policies[policy_id](
      out0,
      out1,
      out2,
      a,
      b,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      num_kv_head,
      group,
      head_dim);
}

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
    const bool is_b_row_major) {
  assert(policy_id >= 0); // should checked outside
  return hgemm_qkv_group_bias_policies[policy_id](
      out0,
      out1,
      out2,
      a,
      b,
      bias,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      num_kv_head,
      group,
      head_dim);
}

} // namespace torch_ipex::xpu::xetla
#endif
