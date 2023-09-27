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

inline __m512 _maskz_load_f32_data(const at::Half* data_base, __mmask16 mask) {
  return cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

// below is for unaligned data load
inline __m512 _loadu(const float* data_base) {
  return _mm512_loadu_ps(data_base);
}

inline __m512 _loadu(const at::BFloat16* data_base) {
  return cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)data_base));
}

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline __m512h _loadu_half(const at::Half* data_base) {
  return _mm512_loadu_ph(data_base);
}
#endif

inline __m512 _loadu(const at::Half* data_base) {
  return cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)data_base));
}

inline __m512 _maskz_loadu(const float* data_base, __mmask16 mask) {
  return _mm512_maskz_loadu_ps(mask, data_base);
}

inline __m512 _maskz_loadu(const at::BFloat16* data_base, __mmask16 mask) {
  return cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline __m512h _maskz_loadu(const at::Half* data_base, __mmask32 mask) {
  return (__m512h)(_mm512_maskz_loadu_epi16(mask, (__m512i*)data_base));
}
#endif

inline __m512 _maskz_loadu(const at::Half* data_base, __mmask16 mask) {
  return cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

// below is for aligned data store
inline void _store_data(float* data_base, __m512 a) {
  _mm512_storeu_ps(data_base, a);
}

inline void _store_data(at::BFloat16* data_base, __m512 a) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_storeu_si256((__m256i*)data_base, vec_bf16_out);
}

inline void _store_data(at::Half* data_base, __m512 a) {
  auto vec_fp16_out = cvt_fp32_to_fp16(a);
  _mm256_storeu_si256((__m256i*)data_base, vec_fp16_out);
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

inline void _mask_store_data(at::Half* data_base, __m512 a, __mmask16 mask) {
  auto vec_fp16_out = cvt_fp32_to_fp16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_fp16_out);
}

// below is for unaligned data store
inline void _storeu(float* data_base, __m512 a) {
  _mm512_storeu_ps(data_base, a);
}

inline void _storeu(at::BFloat16* data_base, __m512 a) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_storeu_si256((__m256i*)data_base, vec_bf16_out);
}

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void _storeu_Half(at::Half* data_base, __m512h a) {
  _mm512_storeu_ph(data_base, a);
}
#endif

inline void _storeu(at::Half* data_base, __m512 a) {
  auto vec_fp16_out = cvt_fp32_to_fp16(a);
  _mm256_storeu_si256((__m256i*)data_base, vec_fp16_out);
}

inline void _mask_storeu(float* data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_storeu_ps(data_base, mask, a);
}

inline void _mask_storeu(at::BFloat16* data_base, __m512 a, __mmask16 mask) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_bf16_out);
}

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void _mask_storeu(at::Half* data_base, __m512h a, __mmask32 mask) {
  _mm512_mask_storeu_epi16((__m512i*)data_base, mask, (__m512i)(a));
}
#endif

inline void _mask_storeu(at::Half* data_base, __m512 a, __mmask16 mask) {
  auto vec_fp16_out = cvt_fp32_to_fp16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_fp16_out);
}
