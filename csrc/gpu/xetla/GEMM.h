#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

void hgemm_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

void hgemm_bias_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k);

void hgemm_bias_res_res_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res0,
    const sycl::half* res1,
    const int m,
    const int n,
    const int k);

void hgemm_bias_gelu_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k);

// 8x512_8x16x16_1

void hgemm_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

void hgemm_bias_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k);

void hgemm_bias_res_res_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res0,
    const sycl::half* res1,
    const int m,
    const int n,
    const int k);

void hgemm_bias_gelu_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k);

void hgemm_qkv_8x128_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

} // namespace xetla
} // namespace xpu
