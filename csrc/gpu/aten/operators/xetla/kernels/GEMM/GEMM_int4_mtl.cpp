#include "gemm_int4.h"

namespace xpu {
namespace xetla {
#define HGEMM_WINT4_MTL_IMPL_FUNC(         \
    WG_M,                                  \
    WG_N,                                  \
    SG_M,                                  \
    SG_N,                                  \
    SG_K,                                  \
    DEQUANT_S,                             \
    SLM_KS,                                \
    L3_KS,                                 \
    SYNC_FREQ,                             \
    STAGES,                                \
    ARCH)                                  \
  template void hgemm_wint4<               \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_bias_wint4<          \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* bias,              \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_qkv_wint4<           \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out0,                   \
      sycl::half * out1,                   \
      sycl::half * out2,                   \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_qkv_bias_wint4<      \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out0,                   \
      sycl::half * out1,                   \
      sycl::half * out2,                   \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* bias,              \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_mul_wint4<           \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* mul,               \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_bias_gelu_wint4<     \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* bias,              \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_bias_res_res_wint4<  \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* bias,              \
      const sycl::half* res0,              \
      const sycl::half* res1,              \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_res_wint4<           \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* res,               \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_silu_mul_wint4<      \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* mul,               \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_bias_silu_mul_wint4< \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* bias,              \
      const sycl::half* mul,               \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_bias_add_wint4<      \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      const sycl::half* bias,              \
      const sycl::half* res,               \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);                   \
  template void hgemm_silu_wint4<          \
      sycl::half,                          \
      WG_M,                                \
      WG_N,                                \
      SG_M,                                \
      SG_N,                                \
      SG_K,                                \
      DEQUANT_S,                           \
      SLM_KS,                              \
      L3_KS,                               \
      SYNC_FREQ,                           \
      STAGES,                              \
      ARCH>(                               \
      sycl::queue & queue,                 \
      sycl::half * out,                    \
      const sycl::half* a,                 \
      const uint8_t* b,                    \
      const uint8_t* b_zp,                 \
      const sycl::half* b_scale,           \
      float* acc_ptr,                      \
      uint32_t* cnt_ptr,                   \
      const uint32_t m,                    \
      const uint32_t n,                    \
      const uint32_t k);

#define HGEMM_WINT4_MTL_IMPL_FUNC_GZ(gz)                                     \
  HGEMM_WINT4_MTL_IMPL_FUNC(                                                 \
      1, 128, 1, 16, 16, gz, 8, 1, 0, 1, static_cast<int>(gpu_arch::XeLpg)); \
  HGEMM_WINT4_MTL_IMPL_FUNC(                                                 \
      16, 32, 8, 16, 16, gz, 8, 1, 0, 1, static_cast<int>(gpu_arch::XeLpg));

HGEMM_WINT4_MTL_IMPL_FUNC_GZ(0); // per channel
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(16);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(32);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(64);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(128);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(256);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(512);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(1024);
} // namespace xetla
} // namespace xpu
