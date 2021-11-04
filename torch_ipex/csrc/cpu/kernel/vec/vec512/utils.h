#pragma once

#include "cpu/bf16/vec/vec_type_cvt.h"

inline __m512 _load_f32_data(const float* data_base) {
  return _mm512_load_ps(data_base);
}

inline __m512 _load_f32_data(const at::BFloat16* data_base) {
  return cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)data_base));
}

inline __m512 _maskz_load_f32_data(const float* data_base, __mmask16 mask) {
  return _mm512_maskz_load_ps(mask, data_base);
}

inline __m512 _maskz_load_f32_data(
    const at::BFloat16* data_base,
    __mmask16 mask) {
  return cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

inline void _store_data(float* data_base, __m512 a) {
  _mm512_store_ps(data_base, a);
}

inline void _store_data(at::BFloat16* data_base, __m512 a) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_store_si256((__m256i*)data_base, vec_bf16_out);
}

inline void _mask_store_data(float* data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_store_ps(data_base, mask, a);
}

inline void _mask_store_data(
    at::BFloat16* data_base,
    __m512 a,
    __mmask16 mask) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_bf16_out);
}
