#pragma once
#include <stddef.h>
#include <sycl/sycl.hpp>
#include <xetla_common_types.hpp>
#include "xetla_kernel_api.h"

namespace torch_ipex::xpu::xetla {
template <
    typename scalar_t,
    gpu::xetla::quant_mode q_mode,
    int WG_N,
    int WG_K,
    int SG_N,
    int SG_K,
    int K_STRIDE,
    int DEQUANT_S,
    int ARCH>
XETLA_KERNEL_API cgf_t xetla_dequantize_int4_weight(
    scalar_t* out,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const uint32_t n,
    const uint32_t k);
}
