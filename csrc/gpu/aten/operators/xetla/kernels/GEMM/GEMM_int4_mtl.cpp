#ifdef USE_XETLA_XE_LPG
#include "gemm_int4.h"

namespace xpu::xetla {
#define HGEMM_WINT4_MTL_IMPL_FUNC(                            \
    scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    DEQUANT_S,                                                \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH,                                                     \
    QUANT_MODE)                                               \
  template cgfs_t XETLA_KERNEL_API hgemm_wint4<               \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_bias_wint4<          \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* bias,                                   \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_qkv_wint4<           \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out0,                                        \
      scalar_t * out1,                                        \
      scalar_t * out2,                                        \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_qkv_bias_wint4<      \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out0,                                        \
      scalar_t * out1,                                        \
      scalar_t * out2,                                        \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* bias,                                   \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_mul_wint4<           \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* mul,                                    \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_bias_gelu_wint4<     \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* bias,                                   \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_bias_res_res_wint4<  \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* bias,                                   \
      const scalar_t* res0,                                   \
      const scalar_t* res1,                                   \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_res_wint4<           \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* res,                                    \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_silu_mul_wint4<      \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* mul,                                    \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_bias_silu_mul_wint4< \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* bias,                                   \
      const scalar_t* mul,                                    \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_bias_add_wint4<      \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      const scalar_t* bias,                                   \
      const scalar_t* res,                                    \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);                                      \
  template cgfs_t XETLA_KERNEL_API hgemm_silu_wint4<          \
      scalar_t,                                               \
      WG_M,                                                   \
      WG_N,                                                   \
      SG_M,                                                   \
      SG_N,                                                   \
      SG_K,                                                   \
      DEQUANT_S,                                              \
      SLM_KS,                                                 \
      L3_KS,                                                  \
      SYNC_FREQ,                                              \
      STAGES,                                                 \
      ARCH,                                                   \
      QUANT_MODE>(                                            \
      scalar_t * out,                                         \
      const scalar_t* a,                                      \
      const uint8_t* b,                                       \
      void* b_zp,                                             \
      const scalar_t* b_scale,                                \
      float* acc_ptr,                                         \
      uint32_t* cnt_ptr,                                      \
      const uint32_t m,                                       \
      const uint32_t n,                                       \
      const uint32_t k);

#define HGEMM_WINT4_MTL_INTERFACE_IMPL( \
    WG_M,                               \
    WG_N,                               \
    SG_M,                               \
    SG_N,                               \
    SG_K,                               \
    DEQUANT_S,                          \
    SLM_KS,                             \
    L3_KS,                              \
    SYNC_FREQ,                          \
    STAGES,                             \
    ARCH)                               \
  HGEMM_WINT4_INTERFACE_IMPL(           \
      WG_M,                             \
      WG_N,                             \
      SG_M,                             \
      SG_N,                             \
      SG_K,                             \
      DEQUANT_S,                        \
      SLM_KS,                           \
      L3_KS,                            \
      SYNC_FREQ,                        \
      STAGES,                           \
      ARCH,                             \
      quant_mode::S4_FULLRANGE_NO_ZP)

#define HGEMM_WINT4_MTL_IMPL_FUNC_GZ(gz)                                    \
  HGEMM_WINT4_MTL_IMPL_FUNC(                                                \
      gpu::xetla::fp16,                                                     \
      1,                                                                    \
      128,                                                                  \
      1,                                                                    \
      16,                                                                   \
      16,                                                                   \
      gz,                                                                   \
      8,                                                                    \
      1,                                                                    \
      0,                                                                    \
      1,                                                                    \
      static_cast<int>(gpu_arch::XeLpg),                                    \
      quant_mode::S4_FULLRANGE_NO_ZP);                                      \
  HGEMM_WINT4_MTL_IMPL_FUNC(                                                \
      gpu::xetla::fp16,                                                     \
      16,                                                                   \
      32,                                                                   \
      8,                                                                    \
      16,                                                                   \
      16,                                                                   \
      gz,                                                                   \
      8,                                                                    \
      1,                                                                    \
      0,                                                                    \
      1,                                                                    \
      static_cast<int>(gpu_arch::XeLpg),                                    \
      quant_mode::S4_FULLRANGE_NO_ZP);                                      \
  HGEMM_WINT4_MTL_INTERFACE_IMPL(                                           \
      1, 128, 1, 16, 16, gz, 8, 1, 0, 1, static_cast<int>(gpu_arch::XeLpg)) \
  HGEMM_WINT4_MTL_INTERFACE_IMPL(                                           \
      16, 32, 8, 16, 16, gz, 8, 1, 0, 1, static_cast<int>(gpu_arch::XeLpg))

HGEMM_WINT4_MTL_IMPL_FUNC_GZ(0); // per channel
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(16);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(32);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(64);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(128);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(256);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(512);
HGEMM_WINT4_MTL_IMPL_FUNC_GZ(1024);
} // namespace xpu::xetla
#endif
