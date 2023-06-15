#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

void gemm(
    sycl::queue& queue,
    float* acc,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

} // namespace xetla
} // namespace xpu
