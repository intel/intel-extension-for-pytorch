#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

void hgemm_8x32_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

void hgemm_8x32_8x16x64_8(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

void hgemm_8x32_8x16x64_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

} // namespace xetla
} // namespace xpu
