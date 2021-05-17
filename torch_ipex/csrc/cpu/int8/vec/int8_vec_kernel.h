#include <immintrin.h>

static inline void zero_ker(int32_t *out, int64_t len) {
  int64_t i;
  __m512i zero_512 = _mm512_setzero_si512();
  #pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi32(out + i, mask, zero_512);
  }
}

static inline void zero_ker(int8_t *out, int64_t len) {
  int64_t i;
  __m512i zero_512 = _mm512_setzero_si512();
  #pragma unroll(4)
  for (i = 0; i < len - 63; i += 64) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi8(out + i, mask, zero_512);
  }
}

static inline __attribute__((always_inline))
void move_ker(int64_t *out, const int64_t *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 7 ; i += 8) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512((void*)(out + i), in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi64(mask, in + i);
    _mm512_mask_storeu_epi64(out + i, mask, in0);
  }
}

static inline __attribute__((always_inline))
void move_ker(int32_t *out, const int32_t *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 15 ; i += 16) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi32(mask, in + i);
    _mm512_mask_storeu_epi32(out + i, mask, in0);
  }
}

static inline __attribute__((always_inline))
void move_ker(int8_t *out, const int8_t *in, int64_t len) {
  int64_t i;
  #pragma unroll(2)
  for (i = 0; i < len - 63 ; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    _mm512_mask_storeu_epi8(out + i, mask, in0);
  }
}

static inline void move_ker(int8_t *out, const int32_t *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 15 ; i += 16) {
    auto in0 = _mm512_loadu_si512(in + i);
    auto out0 = _mm512_cvtepi32_epi8(in0);
    _mm_storeu_si128((__m128i*)(out + i), out0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi32(mask, in + i);
    auto out0 = _mm512_cvtepi32_epi8(in0);
    _mm_mask_storeu_epi8(out + i, mask, out0);
  }
}

static inline void add_ker(int8_t *inout, int8_t *in, int64_t len) {
/*
  for (int64_t i = 0; i < len; ++i) {
    inout[i] += in[i];
  }
*/
  int64_t i;
  #pragma unroll(2)
  for (i = 0; i < len - 63 ; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    auto out = _mm512_loadu_si512(inout + i);
    out = _mm512_adds_epi8(out, in0); //add with saturate
    _mm512_storeu_si512(inout + i, out);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    auto out = _mm512_maskz_loadu_epi8(mask, inout + i);
    out = _mm512_adds_epi8(out, in0);
    _mm512_mask_storeu_epi8(inout + i, mask, out);
  }
}

static inline __attribute__((always_inline))
void scale_and_store_int8_128(void * out, const void *in, __m512 scale) {
  auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
  auto in0_2_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 32)));
  auto in0_3_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 48)));
  auto in0_4_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 64)));
  auto in0_5_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 80)));
  auto in0_6_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 96)));
  auto in0_7_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 112)));
  auto in0_0_32f = _mm512_cvt_roundepi32_ps(in0_0_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_1_32f = _mm512_cvt_roundepi32_ps(in0_1_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_2_32f = _mm512_cvt_roundepi32_ps(in0_2_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_3_32f = _mm512_cvt_roundepi32_ps(in0_3_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_4_32f = _mm512_cvt_roundepi32_ps(in0_4_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_5_32f = _mm512_cvt_roundepi32_ps(in0_5_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_6_32f = _mm512_cvt_roundepi32_ps(in0_6_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_7_32f = _mm512_cvt_roundepi32_ps(in0_7_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_0_32f = _mm512_mul_round_ps(in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_1_32f = _mm512_mul_round_ps(in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_2_32f = _mm512_mul_round_ps(in0_2_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_3_32f = _mm512_mul_round_ps(in0_3_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_4_32f = _mm512_mul_round_ps(in0_4_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_5_32f = _mm512_mul_round_ps(in0_5_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_6_32f = _mm512_mul_round_ps(in0_6_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_7_32f = _mm512_mul_round_ps(in0_7_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_0_32i = _mm512_cvt_roundps_epi32(in0_0_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_1_32i = _mm512_cvt_roundps_epi32(in0_1_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_2_32i = _mm512_cvt_roundps_epi32(in0_2_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_3_32i = _mm512_cvt_roundps_epi32(in0_3_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_4_32i = _mm512_cvt_roundps_epi32(in0_4_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_5_32i = _mm512_cvt_roundps_epi32(in0_5_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_6_32i = _mm512_cvt_roundps_epi32(in0_6_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_7_32i = _mm512_cvt_roundps_epi32(in0_7_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
  _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
  _mm_storeu_si128((__m128i*)(out + 32), _mm512_cvtsepi32_epi8(in0_2_32i));
  _mm_storeu_si128((__m128i*)(out + 48), _mm512_cvtsepi32_epi8(in0_3_32i));
  _mm_storeu_si128((__m128i*)(out + 64), _mm512_cvtsepi32_epi8(in0_4_32i));
  _mm_storeu_si128((__m128i*)(out + 80), _mm512_cvtsepi32_epi8(in0_5_32i));
  _mm_storeu_si128((__m128i*)(out + 96), _mm512_cvtsepi32_epi8(in0_6_32i));
  _mm_storeu_si128((__m128i*)(out + 112), _mm512_cvtsepi32_epi8(in0_7_32i));
}

static inline void scale_and_store_int8_64(void * out, const void *in, __m512 scale) {
  auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
  auto in0_2_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 32)));
  auto in0_3_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 48)));
  auto in0_0_32f = _mm512_cvt_roundepi32_ps(in0_0_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_1_32f = _mm512_cvt_roundepi32_ps(in0_1_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_2_32f = _mm512_cvt_roundepi32_ps(in0_2_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_3_32f = _mm512_cvt_roundepi32_ps(in0_3_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_0_32f = _mm512_mul_round_ps(in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_1_32f = _mm512_mul_round_ps(in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_2_32f = _mm512_mul_round_ps(in0_2_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_3_32f = _mm512_mul_round_ps(in0_3_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_0_32i = _mm512_cvt_roundps_epi32(in0_0_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_1_32i = _mm512_cvt_roundps_epi32(in0_1_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_2_32i = _mm512_cvt_roundps_epi32(in0_2_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_3_32i = _mm512_cvt_roundps_epi32(in0_3_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
  _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
  _mm_storeu_si128((__m128i*)(out + 32), _mm512_cvtsepi32_epi8(in0_2_32i));
  _mm_storeu_si128((__m128i*)(out + 48), _mm512_cvtsepi32_epi8(in0_3_32i));
}

static inline void scale_and_store_int8_32(void* out, const void *in, __m512 scale) {
  auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
  auto in0_0_32f = _mm512_cvt_roundepi32_ps(in0_0_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto in0_1_32f = _mm512_cvt_roundepi32_ps(in0_1_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_0_32f = _mm512_mul_round_ps(in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_1_32f = _mm512_mul_round_ps(in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_0_32i = _mm512_cvt_roundps_epi32(in0_0_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_1_32i = _mm512_cvt_roundps_epi32(in0_1_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
  _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
}

static inline void scale_and_store_int8_16(void* out, const void *in, __m512 scale) {
  auto in0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_32f = _mm512_cvt_roundepi32_ps(in0_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_32f = _mm512_mul_round_ps(in0_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_32i = _mm512_cvt_roundps_epi32(in0_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_32i));
}

static inline void scale_and_store_int8_maskz_16(void * out, const void *in, __m512 scale, __mmask8 mask) {
  auto in0 = _mm_maskz_loadu_epi8(mask, in);
  auto in0_32i = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(mask, in));
  auto in0_32f = _mm512_cvt_roundepi32_ps(in0_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_32f = _mm512_mul_round_ps(in0_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  in0_32i = _mm512_cvt_roundps_epi32(in0_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  _mm_mask_storeu_epi8(out, mask, _mm512_cvtsepi32_epi8(in0_32i));
}

static inline __attribute__((always_inline))
void scale_and_move_ker_128(int8_t *out, const int8_t *in, float scale) {
  __m512 scale_vec512 = _mm512_set1_ps(scale);
  scale_and_store_int8_128((void*)out, (const void*)in, scale_vec512);
}

static inline void scale_and_move_ker(int8_t *out, const int8_t *in, float scale, int64_t len) {
  int64_t i;
/*
  for (i = 0; i < len; i ++) {
     int32_t out_i = (int32_t)((float)in[i] * scale);
     out[i] = (out_f >= 127 ? (int8_t)127 : out_f <= -127 ? (int8_t)-127 : (int8_t)(int32_t)out_f);
  }
*/
  __m512 scale_vec512 = _mm512_set1_ps(scale);
  for (i = 0; i < len - 127 ; i += 128) {
    scale_and_store_int8_128((void*)(out + i), (const void*)(in + i), scale_vec512);
  }
  if ((len - i) > 63) {
    scale_and_store_int8_64((void*)(out + i), (const void*)(in + i), scale_vec512);
    i+= 64;
  }
  if ((len - i) > 31) {
    scale_and_store_int8_32((void*)(out + i), (const void*)(in + i), scale_vec512);
    i+= 32;
  }
  if ((len - i) > 15) {
    scale_and_store_int8_16((void*)(out + i), (const void*)(in + i), scale_vec512);
    i += 16;
  }
  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    scale_and_store_int8_maskz_16(out + i, in + i, scale_vec512, mask);
  }
}

static inline __attribute__((always_inline))
__m512i mul_and_sum_s8x128_to_s32x16(const int8_t *a, const int8_t *b) {
  auto a0 = _mm256_loadu_si256((__m256i*)a);
  auto a0_high = _mm256_loadu_si256((__m256i*)(a + 32));
  auto a1 = _mm256_loadu_si256((__m256i*)(a + 64));
  auto a1_high = _mm256_loadu_si256((__m256i*)(a + 96));
  auto b0 = _mm256_loadu_si256((__m256i*)b);
  auto b0_high = _mm256_loadu_si256((__m256i*)(b + 32));
  auto b1 = _mm256_loadu_si256((__m256i*)(b + 64));
  auto b1_high = _mm256_loadu_si256((__m256i*)(b + 96));
  auto a_0_i = _mm512_cvtepi8_epi16(a0);
  auto a_1_i = _mm512_cvtepi8_epi16(a0_high);
  auto a_2_i = _mm512_cvtepi8_epi16(a1);
  auto a_3_i = _mm512_cvtepi8_epi16(a1_high);
  auto b_0_i = _mm512_cvtepi8_epi16(b0);
  auto b_1_i = _mm512_cvtepi8_epi16(b0_high);
  auto b_2_i = _mm512_cvtepi8_epi16(b1);
  auto b_3_i = _mm512_cvtepi8_epi16(b1_high);
  a_0_i = _mm512_madd_epi16(a_0_i, b_0_i);
  a_2_i = _mm512_madd_epi16(a_2_i, b_2_i);
#ifdef AVX512_VNNI
  a_0_i = _mm512_dpwssd_epi32(a_0_i, a_1_i, b_1_i);
  a_2_i = _mm512_dpwssd_epi32(a_2_i, a_3_i, b_3_i);
#else
  a_1_i = _mm512_madd_epi16(a_1_i, b_1_i);
  a_3_i = _mm512_madd_epi16(a_3_i, b_3_i);
  a_0_i = _mm512_add_epi32(a_0_i, a_1_i);
  a_2_i = _mm512_add_epi32(a_2_i, a_3_i);
#endif
  return _mm512_add_epi32(a_0_i, a_2_i);
}

static inline __attribute__((always_inline))
int32_t hadd_s32x16(__m512i& acc_sum) {
  auto ab_256_high = _mm512_extracti32x8_epi32(acc_sum, 1);
  auto ab_256_low = _mm512_castsi512_si256(acc_sum);
  ab_256_low = _mm256_add_epi32(ab_256_low, ab_256_high);

  auto ab_128_high = _mm256_extracti128_si256(ab_256_low, 1);
  auto ab_128_low = _mm256_castsi256_si128(ab_256_low);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);

  ab_128_high = _mm_unpackhi_epi64(ab_128_low, ab_128_low);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);
  ab_128_high = _mm_shuffle_epi32(ab_128_low, 0xe1);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);
  return _mm_cvtsi128_si32(ab_128_low);
}

static inline __attribute__((always_inline))
int8_t hadd_s32x16_with_scale(__m512i& acc_sum, float scale) {
  auto ab_256_high = _mm512_extracti32x8_epi32(acc_sum, 1);
  auto s_simd = _mm_set1_ps(scale);
  auto ab_256_low = _mm512_castsi512_si256(acc_sum);
  ab_256_low = _mm256_add_epi32(ab_256_low, ab_256_high);

  auto ab_128_high = _mm256_extracti128_si256(ab_256_low, 1);
  auto ab_128_low = _mm256_castsi256_si128(ab_256_low);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);

  ab_128_high = _mm_unpackhi_epi64(ab_128_low, ab_128_low);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);
  ab_128_high = _mm_shuffle_epi32(ab_128_low, 0xe1);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);

  auto ab_128_low_f = _mm_cvtepi32_ps(ab_128_low);
  ab_128_low_f = _mm_mul_round_ss(ab_128_low_f, s_simd, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto sum_vec = _mm_cvtps_epi32(ab_128_low_f);
  sum_vec = _mm_cvtsepi32_epi8(sum_vec);
  return (int8_t)_mm_cvtsi128_si32(sum_vec);
}

static inline __attribute__((always_inline))
void hadd_s32x16x16_with_scales(int8_t* outs, __m512i* acc_sums, __m512& scales) {
  auto l0 = _mm512_unpacklo_epi32(acc_sums[0], acc_sums[1]);
  auto l1 = _mm512_unpackhi_epi32(acc_sums[0], acc_sums[1]);
  auto l2 = _mm512_unpacklo_epi32(acc_sums[2], acc_sums[3]);
  auto l3 = _mm512_unpackhi_epi32(acc_sums[2], acc_sums[3]);
  auto l4 = _mm512_unpacklo_epi32(acc_sums[4], acc_sums[5]);
  auto l5 = _mm512_unpackhi_epi32(acc_sums[4], acc_sums[5]);
  auto l6 = _mm512_unpacklo_epi32(acc_sums[6], acc_sums[7]);
  auto l7 = _mm512_unpackhi_epi32(acc_sums[6], acc_sums[7]);
  l0 = _mm512_add_epi32(l0, l1);
  l2 = _mm512_add_epi32(l2, l3);
  l4 = _mm512_add_epi32(l4, l5);
  l6 = _mm512_add_epi32(l6, l7);
  l1 = _mm512_unpacklo_epi64(l0,l2);
  l3 = _mm512_unpackhi_epi64(l0,l2);
  l5 = _mm512_unpacklo_epi64(l4,l6);
  l7 = _mm512_unpackhi_epi64(l4,l6);
  l1 = _mm512_add_epi32(l1, l3);
  l5 = _mm512_add_epi32(l5, l7);
  l0 = _mm512_shuffle_i32x4(l1, l5, 0x88);
  l2 = _mm512_shuffle_i32x4(l1, l5, 0xdd);
  l0 = _mm512_add_epi32(l0, l2);

  auto h0 = _mm512_unpacklo_epi32(acc_sums[8], acc_sums[9]);
  auto h1 = _mm512_unpackhi_epi32(acc_sums[8], acc_sums[9]);
  auto h2 = _mm512_unpacklo_epi32(acc_sums[10], acc_sums[11]);
  auto h3 = _mm512_unpackhi_epi32(acc_sums[10], acc_sums[11]);
  auto h4 = _mm512_unpacklo_epi32(acc_sums[12], acc_sums[13]);
  auto h5 = _mm512_unpackhi_epi32(acc_sums[12], acc_sums[13]);
  auto h6 = _mm512_unpacklo_epi32(acc_sums[14], acc_sums[15]);
  auto h7 = _mm512_unpackhi_epi32(acc_sums[14], acc_sums[15]);
  h0 = _mm512_add_epi32(h0, h1);
  h2 = _mm512_add_epi32(h2, h3);
  h4 = _mm512_add_epi32(h4, h5);
  h6 = _mm512_add_epi32(h6, h7);
  h1 = _mm512_unpacklo_epi64(h0,h2);
  h3 = _mm512_unpackhi_epi64(h0,h2);
  h5 = _mm512_unpacklo_epi64(h4,h6);
  h7 = _mm512_unpackhi_epi64(h4,h6);
  h1 = _mm512_add_epi32(h1, h3);
  h5 = _mm512_add_epi32(h5, h7);
  h0 = _mm512_shuffle_i32x4(h1, h5, 0x88);
  h2 = _mm512_shuffle_i32x4(h1, h5, 0xdd);
  h0 = _mm512_add_epi32(h0, h2);

  l1 = _mm512_shuffle_i32x4(l0, h0, 0x88);
  h1 = _mm512_shuffle_i32x4(l0, h0, 0xdd);
  l1 = _mm512_add_epi32(l1, h1);
  auto l1_f = _mm512_cvtepi32_ps(l1);
  l1_f = _mm512_mul_round_ps(l1_f, scales, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  l1 = _mm512_cvtps_epi32(l1_f);
  auto out_16 = _mm512_cvtsepi32_epi8(l1);
  _mm_storeu_si128((__m128i*)outs, out_16);
}

static inline __attribute__((always_inline))
void mul_and_sum_s8x128_to_s32x16_aligned_store(__m512i * sum, const int8_t *a, const int8_t *b) {
  auto acc_sum = mul_and_sum_s8x128_to_s32x16(a, b);
  _mm512_store_si512((void*)sum, acc_sum);
}

static inline __attribute__((always_inline))
int32_t mul_and_sum_int8_128(const int8_t *a, const int8_t *b) {
  auto acc_sum = mul_and_sum_s8x128_to_s32x16(a, b);
  return hadd_s32x16(acc_sum);
}

static inline __attribute__((always_inline))
int32_t mul_and_sum_int8_64(const int8_t *a, const int8_t *b) {
  int32_t sum;
  auto a_0_16i = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)a));
  auto a_1_16i = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)(a + 32)));
  auto b_0_16i = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)b));
  auto b_1_16i = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)(b + 32)));
  auto ab_0_32i = _mm512_madd_epi16(a_0_16i, b_0_16i);
  auto ab_1_32i = _mm512_madd_epi16(a_1_16i, b_1_16i);
  ab_0_32i = _mm512_add_epi32(ab_0_32i, ab_1_32i);
  auto ab_256_high = _mm512_extracti32x8_epi32(ab_0_32i, 1);
  auto ab_256_low = _mm512_castsi512_si256(ab_0_32i);
  ab_256_low = _mm256_add_epi32(ab_256_low, ab_256_high);
  auto ab_128_high = _mm256_extracti128_si256(ab_256_low, 1);
  auto ab_128_low = _mm256_castsi256_si128(ab_256_low);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);
  ab_128_high = _mm_unpackhi_epi64(ab_128_low, ab_128_low);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);
  ab_128_high = _mm_shuffle_epi32(ab_128_low, 0xe1);
  ab_128_low = _mm_add_epi32(ab_128_low, ab_128_high);
  sum = _mm_cvtsi128_si32(ab_128_low);
  return sum;
}

static inline __attribute__((always_inline))
int32_t _scale_int32(int32_t value, float scale) {
  auto v_simd = _mm_setzero_ps();
  auto s_simd = _mm_set1_ps(scale);
  v_simd = _mm_cvt_si2ss(v_simd, value);
  v_simd = _mm_mul_round_ss(v_simd, s_simd, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  int32_t c = _mm_cvt_roundss_si32(v_simd, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  auto c_simd =_mm_set1_epi32(c);
  c_simd = _mm_cvtsepi32_epi8(c_simd);
  c = _mm_cvtsi128_si32(c_simd);
  return c;
}

static inline __attribute__((always_inline))
int8_t _dot_s8s8_scale_s32s8(const int8_t* a, const int8_t* b, size_t len, float scale) {
  int32_t c = 0;
  int64_t i;
  for (i = 0; i < len - 127 ; i += 128) {
    c += mul_and_sum_int8_128(a + i, b + i);
  }
  if ((len - i) > 63) {
    c += mul_and_sum_int8_64(a + i, b + i);
    i+= 64;
  }
  for (; i < len; i++) {
    c += (int32_t)a[i] * (int32_t)b[i];
  }
  c = _scale_int32(c, scale);
  return (int8_t)c;
}
