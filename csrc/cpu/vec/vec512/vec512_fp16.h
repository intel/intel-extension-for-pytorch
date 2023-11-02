#pragma once
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec512/vec512.h>
using namespace at::vec;

#include <immintrin.h>
//covert mm512_fp32 to fp8e5m2(__m128i)
inline __m128i _mm512_cvtps_fp8e5m2(__m512 fp32) {
   const __m256i vnaninf = _mm256_set1_epi16 (0x7c00);
   const __m256i vrneadd = _mm256_set1_epi16 (0x007f);
   const __m256i vfixup = _mm256_set1_epi16 (0x0001);
   const __m256i vfixupmask = _mm256_set1_epi16 (0x0100);
   auto fp16 = _mm512_cvtps_ph(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  //mark the values that are not NaN or Inf
   const __mmask32 maska1_ =
      _mm256_cmp_epi16_mask (_mm256_and_si256 (fp16, vnaninf), vnaninf,_MM_CMPINT_NE);
   const __mmask16 maska2_ =
      _mm256_cmp_epi16_mask (_mm256_and_si256 (fp16, vfixupmask), vfixupmask,_MM_CMPINT_EQ);
   __m256i a_rne_ = _mm256_mask_add_epi16 (fp16, maska2_, fp16, _mm256_mask_add_epi16 (vfixup, maska2_, vfixup,vfixup));
   return _mm256_cvtepi16_epi8 (_mm256_srli_epi16 (a_rne_, 8));

}

//covert fp8(__m28i) to fp16(__m256i)
inline __m256i _mm256_cvte5m2_fp16 (__m128i a) {
   return _mm256_slli_epi16 (_mm256_cvtepi8_epi16 (a), 8);
}

//covert fp8e5m2(__m128i) to fp32
inline __m512 _mm512_cvtfp8e5m2_ps(__m128i fp8) {
    auto fp16 = _mm256_cvte5m2_fp16(fp8);
    return _mm512_cvtph_ps(fp16);
}

namespace torch_ipex {
namespace cpu {
namespace kernel {
    template <>
inline __attribute__((always_inline)) void move_ker(
    at::Float8_e5m2* out,
    const at::BFloat16* in,
    int64_t len) {
    int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
      auto fp32 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
      auto fp8e5m2 = _mm512_cvtps_fp8e5m2(fp32);
      _mm_storeu_si128((__m128i*)(out + i), fp8e5m2);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto fp32 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    auto fp8e5m2 = _mm512_cvtps_fp8e5m2(fp32);
    _mm_mask_storeu_epi8((__m128i*)(out + i), mask, fp8e5m2);
  }
}

template <>
inline __attribute__((always_inline)) void move_ker(
    at::Float8_e5m2* out,
    const float* in,
    int64_t len) {
    int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
      auto fp32 = _mm512_loadu_ps(in + i);
      auto fp8e5m2 = _mm512_cvtps_fp8e5m2(fp32);
      _mm_storeu_si128((__m128i*)(out + i), fp8e5m2);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto fp32 = _mm512_maskz_loadu_ps(mask, in + i);
    auto fp8e5m2 = _mm512_cvtps_fp8e5m2(fp32);
    _mm_mask_storeu_epi8((__m128i*)(out + i), mask, fp8e5m2);
  }
}

}
}
}