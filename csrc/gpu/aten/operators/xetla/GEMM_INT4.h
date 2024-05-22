#pragma once
#include <stddef.h>
#include <xetla_common_types.hpp>
#include "xetla_kernel_api.h"

namespace xpu::xetla {
template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_bias_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_bias_gelu_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_mul_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_silu_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_bias_res_res_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    void* res0,
    void* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_qkv_wint4(
    XetlaType xe_type,
    void* out0,
    void* out1,
    void* out2,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_qkv_bias_wint4(
    XetlaType xe_type,
    void* out0,
    void* out1,
    void* out2,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 64,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 16,
    int DQUANT_S = 64,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 0,
    int STAGES = 0,
    int ARCH = 1,
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_silu_mul_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 64,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 16,
    int DQUANT_S = 64,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 0,
    int STAGES = 0,
    int ARCH = 1,
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_bias_silu_mul_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    void* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 64,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 16,
    int DQUANT_S = 64,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 0,
    int STAGES = 0,
    int ARCH = 1,
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_bias_add_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    void* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc),
    int QUANT_MODE = static_cast<int>(quant_mode::S4_FULLRANGE_NO_ZP)>
XETLA_KERNEL_API cgfs_t hgemm_res_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

} // namespace xpu::xetla
