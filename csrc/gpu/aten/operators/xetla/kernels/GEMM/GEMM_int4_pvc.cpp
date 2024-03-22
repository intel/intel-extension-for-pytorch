#ifdef USE_XETLA_XE_HPC
#include "gemm_int4.h"

namespace torch_ipex::xpu::xetla {

#define HGEMM_WINT4_PVC_IMPL_FUNC_ASTR(                       \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      1,                                                      \
      3,                                                      \
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
      const uint32_t k);

#define HGEMM_WINT4_PVC_INTERFACE_IMPL(                    \
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
      1,                                                   \
      3,                                                   \
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
      1,                                                   \
      3,                                                   \
      ARCH,                                                \
      quant_mode::S4_ASYM_ZERO_NO_DEGRAD)

#define HGEMM_WINT4_PVC_IMPL_FUNC(                         \
    WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS, ARCH) \
  HGEMM_WINT4_PVC_IMPL_FUNC_ASTR(                          \
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
  HGEMM_WINT4_PVC_IMPL_FUNC_ASTR(                          \
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
  HGEMM_WINT4_PVC_IMPL_FUNC_ASTR(                          \
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
  HGEMM_WINT4_PVC_IMPL_FUNC_ASTR(                          \
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
  HGEMM_WINT4_PVC_INTERFACE_IMPL(                          \
      WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS, ARCH)

// per channel PVC
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    0,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    0,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    0,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    0,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    0,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    0,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    0,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    0,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    0,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    0,
    1,
    static_cast<int>(gpu_arch::XeHpc));

// k group PVC
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    16,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    16,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    16,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    16,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    16,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    16,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    16,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    16,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    16,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    16,
    1,
    static_cast<int>(gpu_arch::XeHpc));

HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    32,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    32,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    32,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    32,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    32,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    32,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    32,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    32,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    32,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    32,
    1,
    static_cast<int>(gpu_arch::XeHpc));

HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    64,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    64,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    64,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    64,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    64,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    64,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    64,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    64,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    64,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    64,
    1,
    static_cast<int>(gpu_arch::XeHpc));

HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    128,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    128,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    128,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    128,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    128,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    128,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    128,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    128,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    128,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    128,
    1,
    static_cast<int>(gpu_arch::XeHpc));

HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    256,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    256,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    256,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    256,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    256,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    256,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    256,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    256,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    256,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    256,
    1,
    static_cast<int>(gpu_arch::XeHpc));

HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    512,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    512,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    512,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    512,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    512,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    512,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    512,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    512,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    512,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    512,
    1,
    static_cast<int>(gpu_arch::XeHpc));

HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    256,
    8,
    16,
    32,
    1024,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    64,
    8,
    16,
    64,
    1024,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    8,
    512,
    8,
    16,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    256,
    16,
    16,
    32,
    1024,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    64,
    16,
    16,
    32,
    1024,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    16,
    512,
    16,
    16,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    256,
    32,
    16,
    32,
    1024,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    64,
    32,
    16,
    32,
    1024,
    8,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    128,
    32,
    16,
    32,
    1024,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    32,
    512,
    32,
    16,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    256,
    64,
    16,
    32,
    1024,
    2,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    128,
    64,
    16,
    32,
    1024,
    4,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    64,
    512,
    64,
    16,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    256,
    64,
    16,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    128,
    512,
    64,
    32,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));
HGEMM_WINT4_PVC_IMPL_FUNC(
    256,
    256,
    64,
    32,
    32,
    1024,
    1,
    static_cast<int>(gpu_arch::XeHpc));

} // namespace torch_ipex::xpu::xetla
#endif
