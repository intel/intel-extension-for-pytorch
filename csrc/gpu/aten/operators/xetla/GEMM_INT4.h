#pragma once

#include <common/core/common.hpp>
#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {
template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_bias_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_bias_gelu_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_mul_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_silu_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_bias_res_res_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* res0,
    const scalar_t* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_qkv_wint4(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_qkv_bias_wint4(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = 0>
void hgemm_silu_mul_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = 0>
void hgemm_bias_silu_mul_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = 0>
void hgemm_bias_add_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename scalar_t,
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
    int ARCH = static_cast<int>(gpu::xetla::gpu_arch::XeHpc)>
void hgemm_res_wint4(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

} // namespace xetla
} // namespace xpu
