#include "../../hgemm.h"
#include "hgemm_impl.h"

namespace xpu {
namespace xetla {

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

} // namespace xetla
} // namespace xpu
