#include "../../hgemm.h"
#include "hgemm_impl.h"

namespace xpu {
namespace xetla {

#define HGEMM_ENUMERATE_FUNC_IMPLS(                                            \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)                         \
  void HGEMM_ADDMM_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(    \
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
  void HGEMM_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(          \
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
  void HGEMM_BIAS_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(     \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
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
        B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k);                         \
  }                                                                            \
  void HGEMM_BIAS_RES_RES_FUNC(                                                \
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
      const int k) {                                                           \
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
        B_ROW_MAJOR>(queue, out, a, b, bias, res0, res1, m, n, k);             \
  }                                                                            \
  void HGEMM_BIAS_GELU_FUNC(                                                   \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
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
        B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k);                         \
  }                                                                            \
  void HGEMM_RESMUL_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(   \
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
  void HGEMM_SILU_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(     \
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
  void HGEMM_RES_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* res,                                                   \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
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
        B_ROW_MAJOR>(queue, out, a, b, res, m, n, k);                          \
  }                                                                            \
  void HGEMM_QKV_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(      \
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
  void HGEMM_QKV_BIAS_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
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
  }                                                                            \
  void HGEMM_BIAS_RES_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res,                                                   \
      const sycl::half res_scale,                                              \
      const int m,                                                             \
      const int n,                                                             \
      const int k) {                                                           \
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
        B_ROW_MAJOR>(queue, out, a, b, bias, res, res_scale, m, n, k);         \
  }

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_FUNC_IMPLS)

} // namespace xetla
} // namespace xpu
