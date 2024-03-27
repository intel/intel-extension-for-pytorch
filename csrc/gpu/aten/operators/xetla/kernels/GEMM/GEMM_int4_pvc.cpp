#include "gemm_int4.h"

namespace xpu {
namespace xetla {

#define HGEMM_WINT4_PVC_IMPL_FUNC(                         \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
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
      const uint32_t k);

// per channel PVC
HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 0, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 0, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 0, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 0, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 0, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 0, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 0, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 0, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 0, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 0, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 0, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 0, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 0, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 0, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 0, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 0, 1, 1);

// k group PVC
HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 16, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 16, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 16, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 16, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 16, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 16, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 16, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 16, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 16, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 16, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 16, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 16, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 16, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 16, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 16, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 16, 1, 1);

HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 32, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 32, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 32, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 32, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 32, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 32, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 32, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 32, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 32, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 32, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 32, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 32, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 32, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 32, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 32, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 32, 1, 1);

HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 64, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 64, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 64, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 64, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 64, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 64, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 64, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 64, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 64, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 64, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 64, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 64, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 64, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 64, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 64, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 64, 1, 1);

HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 128, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 128, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 128, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 128, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 128, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 128, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 128, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 128, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 128, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 128, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 128, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 128, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 128, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 128, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 128, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 128, 1, 1);

HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 256, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 256, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 256, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 256, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 256, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 256, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 256, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 256, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 256, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 256, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 256, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 256, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 256, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 256, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 256, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 256, 1, 1);

HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 512, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 512, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 512, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 512, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 512, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 512, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 512, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 512, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 512, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 512, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 512, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 512, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 512, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 512, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 512, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 512, 1, 1);

HGEMM_WINT4_PVC_IMPL_FUNC(8, 256, 8, 16, 32, 1024, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 64, 8, 16, 64, 1024, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(8, 512, 8, 16, 32, 1024, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 256, 16, 16, 32, 1024, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 64, 16, 16, 32, 1024, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(16, 512, 16, 16, 32, 1024, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 256, 32, 16, 32, 1024, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 64, 32, 16, 32, 1024, 8, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 128, 32, 16, 32, 1024, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(32, 512, 32, 16, 32, 1024, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 256, 64, 16, 32, 1024, 2, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 128, 64, 16, 32, 1024, 4, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(64, 512, 64, 16, 32, 1024, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 256, 64, 16, 32, 1024, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(128, 512, 64, 32, 32, 1024, 1, 1);
HGEMM_WINT4_PVC_IMPL_FUNC(256, 256, 64, 32, 32, 1024, 1, 1);

} // namespace xetla
} // namespace xpu
