#include "gemm_int4.h"

namespace xpu {
namespace xetla {

#define HGEMM_WINT4_ARC_IMPL_FUNC(                         \
    WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS, ARCH) \
  template void hgemm_wint4<                               \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_bias_wint4<                          \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* bias,                              \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_qkv_wint4<                           \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out0,                                   \
      sycl::half * out1,                                   \
      sycl::half * out2,                                   \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_qkv_bias_wint4<                      \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out0,                                   \
      sycl::half * out1,                                   \
      sycl::half * out2,                                   \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* bias,                              \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_mul_wint4<                           \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* mul,                               \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_bias_gelu_wint4<                     \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* bias,                              \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_bias_res_res_wint4<                  \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* bias,                              \
      const sycl::half* res0,                              \
      const sycl::half* res1,                              \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_res_wint4<                           \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* res,                               \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_silu_mul_wint4<                      \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* mul,                               \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_bias_silu_mul_wint4<                 \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* bias,                              \
      const sycl::half* mul,                               \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_bias_add_wint4<                      \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      const sycl::half* bias,                              \
      const sycl::half* res,                               \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);                                   \
  template void hgemm_silu_wint4<                          \
      sycl::half,                                          \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      1,                                                   \
      0,                                                   \
      0,                                                   \
      ARCH>(                                               \
      sycl::queue & queue,                                 \
      sycl::half * out,                                    \
      const sycl::half* a,                                 \
      const uint8_t* b,                                    \
      const uint8_t* b_zp,                                 \
      const sycl::half* b_scale,                           \
      float* acc_ptr,                                      \
      uint32_t* cnt_ptr,                                   \
      const uint32_t m,                                    \
      const uint32_t n,                                    \
      const uint32_t k);

// per channel ARC
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 0, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 0, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 0, 1, 0);

// k group ARC
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 16, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 32, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 64, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 128, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 256, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 512, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 1024, 8, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 16, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 32, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 64, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 128, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 256, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 512, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 1024, 4, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 16, 1, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 32, 1, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 64, 1, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 128, 1, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 256, 1, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 512, 1, 0);
HGEMM_WINT4_ARC_IMPL_FUNC(32, 256, 16, 16, 32, 1024, 1, 0);
} // namespace xetla
} // namespace xpu