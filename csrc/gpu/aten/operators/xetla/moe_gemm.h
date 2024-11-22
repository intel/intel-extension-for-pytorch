#pragma once

#include <sycl/sycl.hpp>
#include "xetla_kernel_api.h"

namespace gpu {
namespace xetla {
using cgfs_t = torch_ipex::xpu::xetla::cgfs_t;

template <typename T>
XETLA_KERNEL_API cgfs_t moe_gemm(
    sycl::queue& queue,
    const T* activations,
    const T* weights,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count);

} // namespace xetla
} // namespace gpu