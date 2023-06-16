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

// m == 1 && n == 4096 && k == 4096
void hgemm_8x128_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

// m == 1 && n == 4096 && k == 16384
void hgemm_bias_8x128_8x16x16_4(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k);

// m == 1 && n == 16384 && k == 4096
// m == 1 && n == 32000 && k == 4096
void hgemm_bias_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k);

} // namespace xetla
} // namespace xpu
