#include "../../hgemm.h"
#include "hgemm_impl.h"
#include "hgemm_policy.h"

namespace xpu {
namespace xetla {

// clang-format off

#define HGEMM_IMPL_NAME(HEAD, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HEAD##_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

#define HGEMM_ADDMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_addmm, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_COMMON_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_common, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_BIAS_GELU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_gelu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RESMUL_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_resmul, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_SILU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_silu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_QKV_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

// clang-format on

#define HGEMM_ENUMERATE_IMPLS(                                                 \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)                         \
  void HGEMM_ADDMM_IMPL_NAME(                                                  \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* res,                                                   \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float alpha,                                                       \
      const float beta) {                                                      \
    hgemm_addmm<                                                               \
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
        B_ROW_MAJOR>(queue, out, res, a, b, m, n, k, alpha, beta);             \
  }                                                                            \
  void HGEMM_COMMON_IMPL_NAME(                                                 \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    hgemm_common<                                                              \
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
        B_ROW_MAJOR>(queue, out, a, b, m, n, k);                               \
  }                                                                            \
  void HGEMM_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* res,                                                   \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float res_factor) {                                                \
    hgemm_res<                                                                 \
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
        B_ROW_MAJOR>(queue, out, a, b, res, m, n, k, res_factor);              \
  }                                                                            \
  void HGEMM_RES_RES_IMPL_NAME(                                                \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* res0,                                                  \
      const sycl::half* res1,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float res0_factor,                                                 \
      const float res1_factor) {                                               \
    hgemm_res_res<                                                             \
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
        queue, out, a, b, res0, res1, m, n, k, res0_factor, res1_factor);      \
  }                                                                            \
  void HGEMM_BIAS_IMPL_NAME(                                                   \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor) {                                               \
    hgemm_bias<                                                                \
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
        B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k, bias_factor);            \
  }                                                                            \
  void HGEMM_BIAS_RES_IMPL_NAME(                                               \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res,                                                   \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor,                                                 \
      const float res_factor) {                                                \
    hgemm_bias_res<                                                            \
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
        queue, out, a, b, bias, res, m, n, k, bias_factor, res_factor);        \
  }                                                                            \
  void HGEMM_BIAS_RES_RES_IMPL_NAME(                                           \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res0,                                                  \
      const sycl::half* res1,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor,                                                 \
      const float res0_factor,                                                 \
      const float res1_factor) {                                               \
    hgemm_bias_res_res<                                                        \
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
        queue,                                                                 \
        out,                                                                   \
        a,                                                                     \
        b,                                                                     \
        bias,                                                                  \
        res0,                                                                  \
        res1,                                                                  \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        bias_factor,                                                           \
        res0_factor,                                                           \
        res1_factor);                                                          \
  }                                                                            \
  void HGEMM_BIAS_GELU_IMPL_NAME(                                              \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float bias_factor) {                                               \
    hgemm_bias_gelu<                                                           \
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
        B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k, bias_factor);            \
  }                                                                            \
  void HGEMM_RESMUL_IMPL_NAME(                                                 \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* mul,                                                   \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    hgemm_mul<                                                                 \
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
        B_ROW_MAJOR>(queue, out, a, b, mul, m, n, k);                          \
  }                                                                            \
  void HGEMM_SILU_IMPL_NAME(                                                   \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    hgemm_silu<                                                                \
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
        B_ROW_MAJOR>(queue, out, a, b, m, n, k);                               \
  }                                                                            \
  void HGEMM_QKV_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue,                                                     \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    hgemm_qkv<                                                                 \
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
        B_ROW_MAJOR>(queue, out0, out1, out2, a, b, m, n, k);                  \
  }                                                                            \
  void HGEMM_QKV_BIAS_IMPL_NAME(                                               \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
    hgemm_qkv_bias<                                                            \
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
        B_ROW_MAJOR>(queue, out0, out1, out2, a, b, bias, m, n, k);            \
  }

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_IMPLS)

void (*hgemm_addmm_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_ADDMM_IMPL_NAME)};

void (*hgemm_common_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_COMMON_IMPL_NAME)};

void (*hgemm_res_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_IMPL_NAME)};

void (*hgemm_res_res_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_RES_IMPL_NAME)};

void (*hgemm_bias_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_IMPL_NAME)};

void (*hgemm_bias_res_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_IMPL_NAME)};

void (*hgemm_bias_res_res_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float,
    const float,
    const float) = {
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_RES_IMPL_NAME)};

void (*hgemm_bias_gelu_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_GELU_IMPL_NAME)};

void (*hgemm_resmul_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RESMUL_IMPL_NAME)};

void (*hgemm_silu_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_SILU_IMPL_NAME)};

void (*hgemm_qkv_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_IMPL_NAME)};

void (*hgemm_qkv_bias_policies[HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_BIAS_IMPL_NAME)};

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_addmm_policies[policy_id](queue, out, res, a, b, m, n, k, alpha, beta);
  return GemmStatus::kSuccess;
}

GemmStatus hgemm_common(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_common_policies[policy_id](queue, out, a, b, m, n, k);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_res_policies[policy_id](queue, out, a, b, res, m, n, k, res_factor);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_res_res_policies[policy_id](
      queue, out, a, b, res0, res1, m, n, k, res0_factor, res1_factor);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_bias_policies[policy_id](queue, out, a, b, bias, m, n, k, bias_factor);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_bias_res_policies[policy_id](
      queue, out, a, b, bias, res, m, n, k, bias_factor, res_factor);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_bias_res_res_policies[policy_id](
      queue,
      out,
      a,
      b,
      bias,
      res0,
      res1,
      m,
      n,
      k,
      bias_factor,
      res0_factor,
      res1_factor);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_bias_gelu_policies[policy_id](
      queue, out, a, b, bias, m, n, k, bias_factor);
  return GemmStatus::kSuccess;
}

GemmStatus hgemm_resmul(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* mul,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_resmul_policies[policy_id](queue, out, a, b, mul, m, n, k);
  return GemmStatus::kSuccess;
}

GemmStatus hgemm_silu(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major) {
  int policy_id = hgemm_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_silu_policies[policy_id](queue, out, a, b, m, n, k);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_qkv_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_qkv_policies[policy_id](queue, out0, out1, out2, a, b, m, n, k);
  return GemmStatus::kSuccess;
}

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
    const bool is_b_row_major) {
  int policy_id = hgemm_qkv_find_policy_id(m, n, k, is_b_row_major);
  if (policy_id < 0)
    return GemmStatus::kError;
  hgemm_qkv_bias_policies[policy_id](
      queue, out0, out1, out2, a, b, bias, m, n, k);
  return GemmStatus::kSuccess;
}

} // namespace xetla
} // namespace xpu
