#pragma once

// below is for aligned data load
inline __m512 _load_f32_data(const float* data_base) {
  return _mm512_loadu_ps(data_base);
}

inline __m512 _load_f32_data(const at::BFloat16* data_base) {
  return cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)data_base));
}

inline __m512 _maskz_load_f32_data(const float* data_base, __mmask16 mask) {
  return _mm512_maskz_loadu_ps(mask, data_base);
}

inline __m512 _maskz_load_f32_data(
    const at::BFloat16* data_base,
    __mmask16 mask) {
  return cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

// below is for unaligned data load
inline __m512 _loadu(const float* data_base) {
  return _mm512_loadu_ps(data_base);
}

inline __m512 _loadu(const at::BFloat16* data_base) {
  return cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)data_base));
}

inline __m512 _maskz_loadu(const float* data_base, __mmask16 mask) {
  return _mm512_maskz_loadu_ps(mask, data_base);
}

inline __m512 _maskz_loadu(const at::BFloat16* data_base, __mmask16 mask) {
  return cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

// below is for aligned data store
inline void _store_data(float* data_base, __m512 a) {
  _mm512_storeu_ps(data_base, a);
}

inline void _store_data(at::BFloat16* data_base, __m512 a) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_storeu_si256((__m256i*)data_base, vec_bf16_out);
}

inline void _mask_store_data(float* data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_storeu_ps(data_base, mask, a);
}

inline void _mask_store_data(
    at::BFloat16* data_base,
    __m512 a,
    __mmask16 mask) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_bf16_out);
}

// below is for unaligned data store
inline void _storeu(float* data_base, __m512 a) {
  _mm512_storeu_ps(data_base, a);
}

inline void _storeu(at::BFloat16* data_base, __m512 a) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_storeu_si256((__m256i*)data_base, vec_bf16_out);
}

inline void _mask_storeu(float* data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_storeu_ps(data_base, mask, a);
}

inline void _mask_storeu(at::BFloat16* data_base, __m512 a, __mmask16 mask) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_bf16_out);
}

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline __m512 _loadu(const at::Float8_e5m2* data_base) {
  return _mm512_cvtfp8e5m2_ps(_mm_loadu_si128((__m128i*)data_base));
}

inline __m512 _maskz_loadu(const at::Float8_e5m2* data_base, __mmask16 mask) {
  return _mm512_cvtfp8e5m2_ps(_mm_maskz_loadu_epi16(mask, (__m128i*)data_base));
}

inline void _storeu(at::Float8_e5m2* data_base, __m512 a) {
  auto vec_fp8_out = _mm512_cvtps_fp8e5m2(a);
  _mm_storeu_si128((__m128i*)data_base, vec_fp8_out);
}

inline void _mask_storeu(at::Float8_e5m2* data_base, __m512 a, __mmask16 mask) {
  auto vec_fp8_out = _mm512_cvtps_fp8e5m2(a);
  _mm_mask_storeu_epi8(data_base, mask, vec_fp8_out);
}
#else
inline __m512 _loadu(const at::Float8_e5m2* data_base) {
  TORCH_INTERNAL_ASSERT(false, "fp8_e5m2 is not supported on this platform");
}

inline __m512 _maskz_loadu(const at::Float8_e5m2* data_base, __mmask16 mask) {
  TORCH_INTERNAL_ASSERT(false, "fp8_e5m2 is not supported on this platform");
}

inline void _storeu(at::Float8_e5m2* data_base, __m512 a) {
  TORCH_INTERNAL_ASSERT(false, "fp8_e5m2 is not supported on this platform");
}

inline void _mask_storeu(at::Float8_e5m2* data_base, __m512 a, __mmask16 mask) {
  TORCH_INTERNAL_ASSERT(false, "fp8_e5m2 is not supported on this platform");
}
#endif