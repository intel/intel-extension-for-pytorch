#pragma once
#include <immintrin.h>
#include <cstdlib>
#include "utils/SysUtil.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {

static IPEX_FORCE_INLINE __m512i _mm512_cvte5m2_fp16(__m256i a) {
  return _mm512_slli_epi16(_mm512_cvtepi8_epi16(a), 8);
}

static IPEX_FORCE_INLINE __m512 _mm512_cvtpbh_ps(__m256i x) {
  return (__m512)_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 0x10);
}

static IPEX_FORCE_INLINE void cvt_fp32_e5m2_rne_intrinsic(
    const float* __restrict__ in,
    at::Float8_e5m2* out,
    int64_t len) {
  int64_t i = 0;
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  for (; i < len - 31; i += 32) {
    __m512 b = _mm512_loadu_ps(&in[i]);
    __m512 a = _mm512_loadu_ps(&in[i + 16]);

    __m256i ah_ =
        _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i bh_ =
        _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m512i a_ = _mm512_inserti64x4(
        _mm512_inserti64x4(_mm512_setzero_si512(), bh_, 0), ah_, 1);
    const __mmask32 maska1_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vnaninf), vnaninf, _MM_CMPINT_NE);
    const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
    __m512i a_rne_ = _mm512_mask_add_epi16(
        a_,
        maska1_,
        a_,
        _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
    a_rne_ = _mm512_srli_epi16(a_rne_, 8);
    _mm256_storeu_epi8(&out[i], _mm512_cvtepi16_epi8(a_rne_));
  }

  for (; i < len; i++) {
    out[i] = static_cast<at::Float8_e5m2>(in[i]);
  }
}

static IPEX_FORCE_INLINE void cvt_bf16_e5m2_rne_intrinsic(
    const at::BFloat16* __restrict__ in,
    at::Float8_e5m2* out,
    int64_t len) {
  int64_t i = 0;
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  for (; i < len - 31; i += 32) {
    __m512i x0 = _mm512_loadu_si512(&in[i]);
    __m512 b = _mm512_cvtpbh_ps(_mm512_extracti32x8_epi32(x0, 0));
    __m512 a = _mm512_cvtpbh_ps(_mm512_extracti32x8_epi32(x0, 1));

    __m256i ah_ =
        _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i bh_ =
        _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m512i a_ = _mm512_inserti64x4(
        _mm512_inserti64x4(_mm512_setzero_si512(), bh_, 0), ah_, 1);
    const __mmask32 maska1_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vnaninf), vnaninf, _MM_CMPINT_NE);
    const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
    __m512i a_rne_ = _mm512_mask_add_epi16(
        a_,
        maska1_,
        a_,
        _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
    a_rne_ = _mm512_srli_epi16(a_rne_, 8);
    _mm256_storeu_epi8(&out[i], _mm512_cvtepi16_epi8(a_rne_));
  }

  for (; i < len; i++) {
    out[i] = static_cast<at::Float8_e5m2>(in[i]);
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex