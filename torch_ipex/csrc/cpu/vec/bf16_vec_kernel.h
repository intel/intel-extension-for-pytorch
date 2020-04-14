#include <immintrin.h>
#include "vec_type_cvt.h"

inline __m512 pack_bf16_to_fp32(const __m256i top, const __m256i bot) {
  auto x1 = _mm512_cvtepu16_epi32(top);
  auto x2 = _mm512_cvtepu16_epi32(bot);
  auto y = _mm512_add_epi32(_mm512_bslli_epi128(x1, 2), x2);
  return _mm512_castsi512_ps(y);
}

// Only support AVX512 impl at current stage. Will expand this impl to cover AVX2 and other cases.
inline void packed_bf16_add_ker(at::BFloat16 *a1, at::BFloat16 *a2, at::BFloat16 *b, int len, float alpha) {
  auto vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto x1 = _mm256_loadu_si256((__m256i *)(a1 + i));
    auto x2 = _mm256_loadu_si256((__m256i *)(a2 + i));
    auto y1 = _mm256_loadu_si256((__m256i *)(b + i));

    auto z1 = pack_bf16_to_fp32(x1, x2);
    auto z2 = cvt_bf16_to_fp32(y1);
    z1 = _mm512_fmadd_ps(vAlpha, z2, z1);
    // Update result back to split input tensors.
    _mm256_storeu_si256((__m256i *)(a1 + i), trunc_fp32_to_bf16(z1));
    _mm256_storeu_si256((__m256i *)(a2 + i), _mm512_cvtepi32_epi16(_mm512_castps_si512(z1)));
  }

  if (i < len) {
    __mmask16 mask = (1 << (len - i)) - 1;
    auto x1 = _mm256_maskz_loadu_epi16(mask, a1 + i);
    auto x2 = _mm256_maskz_loadu_epi16(mask, a2 + i);
    auto y1 = _mm256_maskz_loadu_epi16(mask, b + i);

    auto z1 = pack_bf16_to_fp32(x1, x2);
    auto z2 = cvt_bf16_to_fp32(y1);
    z1 = _mm512_fmadd_ps(vAlpha, z2, z1);
    // Update result back to split input tensors.
    _mm256_mask_storeu_epi16(a1 + i, mask, trunc_fp32_to_bf16(z1));
    _mm256_mask_storeu_epi16(a2 + i, mask, _mm512_cvtepi32_epi16(_mm512_castps_si512(z1)));
  }

}
