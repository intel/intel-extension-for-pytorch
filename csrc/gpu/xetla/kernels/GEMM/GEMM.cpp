#include "../../GEMM.h"
#include "hgemm_splitk.h"

namespace xpu {
namespace xetla {

void hgemm_8x32_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  // m == 1 && n == 4096 && k == 4096
  hgemm_common<sycl::half, 8, 32, 8, 16, 32, 4, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_8x32_8x16x64_8(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  // m == 1 && n == 4096 && k == 16384
  hgemm_common<sycl::half, 8, 32, 8, 16, 64, 8, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_8x32_8x16x64_1(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  // m == 1 && n == 16384 && k == 4096
  // m == 1 && n == 32000 && k == 4096
  hgemm_common<sycl::half, 8, 32, 8, 16, 64, 1, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_32x64_8x16x16_2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  // m == 1 && n == 4096 && k == 4096
  hgemm_common<sycl::half, 32, 64, 8, 16, 16, 2, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_8x128_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k) {
  // m == 1 && n == 4096 && k == 4096
  hgemm_common<sycl::half, 8, 128, 8, 16, 32, 4, 1, 1, 3, true>(
      queue, out, a, b, m, n, k);
}

void hgemm_bias_8x128_8x16x16_4(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const int m,
    const int n,
    const int k) {
  // m == 1 && n == 4096 && k == 16384
  hgemm_bias<sycl::half, 8, 128, 8, 16, 16, 4, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
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
  // m == 1 && n == 4096 && k == 16384
  hgemm_bias<sycl::half, 32, 64, 8, 16, 16, 2, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
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
  // m == 1 && n == 16384 && k == 4096
  // m == 1 && n == 32000 && k == 4096
  hgemm_bias<sycl::half, 8, 512, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
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
  // m == 1 && n == 16384 && k == 4096
  // m == 1 && n == 32000 && k == 4096
  hgemm_bias_gelu<sycl::half, 8, 512, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out, a, b, bias, m, n, k);
}

} // namespace xetla
} // namespace xpu
