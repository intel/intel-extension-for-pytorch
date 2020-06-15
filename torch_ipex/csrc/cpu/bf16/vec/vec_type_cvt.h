#include <immintrin.h>

// Conversion from BF16 to FP32
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline void cvt_bf16_to_fp32(float *dst, const at::BFloat16 *src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i *)(src + i)));
    _mm512_storeu_ps(dst + i, f32);
  }
  if (i < len) {
    auto  mask = (1 << (len - i)) - 1;
    auto f32 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, src + i));
    _mm512_mask_storeu_ps(dst + i, mask, f32);
  }
}

// Conversion from FP32 to BF16
inline __m256i trunc_fp32_to_bf16(const __m512 src) {
  auto y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if defined(AVX512_BF16)
  return _mm512_cvtneps_pbh(src);
#else
  return trunc_fp32_to_bf16(src);
#endif
}

inline void cvt_fp32_to_bf16(at::BFloat16 *dst, const float *src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256((__m256i *)(dst + i), cvt_fp32_to_bf16(f32));
  }
  if (i < len) {
    auto mask = (1 << (len - i )) - 1;
    auto f32 = _mm512_maskz_loadu_ps(mask, src + i);
    _mm256_mask_storeu_epi16(dst + i, mask, cvt_fp32_to_bf16(f32));
  }
}
