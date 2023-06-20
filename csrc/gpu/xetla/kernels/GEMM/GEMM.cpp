#include "../../GEMM.h"
#include "hgemm_splitk.h"

namespace xpu {
namespace xetla {

// 32x64_8x16x16_2

void hgemm_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  hgemm_common<sycl::half, 32, 64, 8, 16, 16, 2, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_bias_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k) {
  hgemm_bias<sycl::half, 32, 64, 8, 16, 16, 2, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
}

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
    const int k) {
  hgemm_bias_res_res<sycl::half, 32, 64, 8, 16, 16, 2, 1, 1, 3, true>(
      queue, out, a, b, bias, res0, res1, m, n, k);
}

void hgemm_bias_gelu_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k) {
  hgemm_bias_gelu<sycl::half, 32, 64, 8, 16, 16, 2, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
}

// 8x512_8x16x16_1

void hgemm_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  hgemm_common<sycl::half, 8, 512, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_bias_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k) {
  hgemm_bias<sycl::half, 8, 512, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
}

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
    const int k) {
  hgemm_bias_res_res<sycl::half, 8, 512, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out, a, b, bias, res0, res1, m, n, k);
}

void hgemm_bias_gelu_8x512_8x16x16_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k) {
  hgemm_bias_gelu<sycl::half, 8, 512, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
}

void hgemm_qkv_8x128_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  hgemm_qkv<sycl::half, 8, 128, 8, 16, 32, 4, 1, 1, 3, true>(
      queue, out0, out1, out2, a, b, m, n, k);
}

} // namespace xetla
} // namespace xpu
