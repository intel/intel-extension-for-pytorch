#ifdef USE_XETLA_XE_HPG
#include "gemm_int4.h"

namespace xpu::xetla {

#define HGEMM_WINT4_ARC_IMPL_FUNC_ASTR(                       \
    scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    DEQUANT_S,                                                \
    SLM_KS,                                                   \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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
      1,                                                      \
      0,                                                      \
      0,                                                      \
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

#define HGEMM_WINT4_ARC_INTERFACE_IMPL(                    \
    WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS, ARCH) \
  HGEMM_WINT4_INTERFACE_IMPL(                              \
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
      ARCH,                                                \
      quant_mode::S4_FULLRANGE_NO_ZP)                      \
  HGEMM_WINT4_INTERFACE_IMPL(                              \
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
      ARCH,                                                \
      quant_mode::S4_ASYM_ZERO_NO_DEGRAD)

#define HGEMM_WINT4_ARC_IMPL_FUNC(                         \
    WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS, ARCH) \
  HGEMM_WINT4_ARC_IMPL_FUNC_ASTR(                          \
      gpu::xetla::fp16,                                    \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      ARCH,                                                \
      quant_mode::S4_FULLRANGE_NO_ZP)                      \
  HGEMM_WINT4_ARC_IMPL_FUNC_ASTR(                          \
      gpu::xetla::fp16,                                    \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      ARCH,                                                \
      quant_mode::S4_ASYM_ZERO_NO_DEGRAD)                  \
  HGEMM_WINT4_ARC_IMPL_FUNC_ASTR(                          \
      gpu::xetla::bf16,                                    \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      ARCH,                                                \
      quant_mode::S4_FULLRANGE_NO_ZP)                      \
  HGEMM_WINT4_ARC_IMPL_FUNC_ASTR(                          \
      gpu::xetla::bf16,                                    \
      WG_M,                                                \
      WG_N,                                                \
      SG_M,                                                \
      SG_N,                                                \
      SG_K,                                                \
      DEQUANT_S,                                           \
      SLM_KS,                                              \
      ARCH,                                                \
      quant_mode::S4_ASYM_ZERO_NO_DEGRAD)                  \
  HGEMM_WINT4_ARC_INTERFACE_IMPL(                          \
      WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS, ARCH)

// per channel ARC
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    0,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    0,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpg));

// k group ARC
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    16,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    32,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    64,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    128,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    256,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    512,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    1024,
    8,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    16,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    32,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    64,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    128,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    256,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    512,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    16,
    1024,
    4,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpg));
HGEMM_WINT4_ARC_IMPL_FUNC(
    32,
    256,
    16,
    16,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpg));
} // namespace xpu::xetla
#endif
