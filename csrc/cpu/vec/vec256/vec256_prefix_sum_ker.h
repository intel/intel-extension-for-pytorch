#pragma once

namespace torch_ipex {
namespace cpu {
namespace kernel {

template <>
inline void prefix_sum<int64_t>(
    const int64_t* src,
    int64_t* dst,
    int64_t init,
    int64_t n) {
  int64_t i;
  __m256i offset = _mm256_set1_epi64x(init);
  __m256i zero = _mm256_setzero_si256();
  for (i = 0; i <= (n - Vectorized<int64_t>::size());
       i += Vectorized<int64_t>::size()) {
    // a = {a0, a1, a2, a3}
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    __m256i x0 = _mm256_permute4x64_epi64(a, 0b10010011);
    x0 = _mm256_blend_epi32(x0, zero, 0b00000011);

    // x1 = {a0, a01, a12, a23}
    // x2 = {0, 0, a0, a01}
    __m256i x1 = _mm256_add_epi64(a, x0);
    __m256i x2 = _mm256_permute2f128_si256(x1, x1, 0b00101000);

    // x1 = {a0, a01, a012, a0123}
    x1 = _mm256_add_epi64(x1, x2);
    __m256i y = _mm256_add_epi64(offset, x1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), y);

    // broadcast offset
    offset = _mm256_permute4x64_epi64(y, 0b11111111);
  }
  int64_t offset_v = i == 0 ? init : dst[i - 1];
  for (; i < n; i++) {
    offset_v += src[i];
    dst[i] = offset_v;
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
