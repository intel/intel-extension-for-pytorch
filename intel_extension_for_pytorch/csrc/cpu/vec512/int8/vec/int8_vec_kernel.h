#pragma once
#include <cstdlib>

#if defined(CPU_AVX512)
#include <immintrin.h>
#else
#include "csrc/cpu/vec512/ref/add_ker.h"
#include "csrc/cpu/vec512/ref/mov_ker.h"
using namespace torch_ipex::cpu::kernel;
#endif

static inline __attribute__((always_inline)) void zero_ker(
    int32_t* out,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
  __m512i zero_512 = _mm512_setzero_si512();
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi32(out + i, mask, zero_512);
  }
#else
  memset(out, 0, sizeof(int32_t) * len);
#endif
}

static inline __attribute__((always_inline)) void zero_ker(
    int8_t* out,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
  __m512i zero_512 = _mm512_setzero_si512();
#pragma unroll(4)
  for (i = 0; i < len - 63; i += 64) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi8(out + i, mask, zero_512);
  }
#else
  memset(out, 0, sizeof(int8_t) * len);
#endif
}

static inline __attribute__((always_inline)) void move_ker(
    int64_t* out,
    const int64_t* in,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(4)
  for (i = 0; i < len - 7; i += 8) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512((void*)(out + i), in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi64(mask, in + i);
    _mm512_mask_storeu_epi64(out + i, mask, in0);
  }
#else
  ref::mov_ker(out, in, len);
#endif
}

static inline __attribute__((always_inline)) void move_ker(
    int16_t* out,
    const int16_t* in,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi16(mask, in + i);
    _mm512_mask_storeu_epi16(out + i, mask, in0);
  }
#else
  ref::mov_ker(out, in, len);
#endif
}

static inline __attribute__((always_inline)) void move_ker(
    unsigned char* out,
    const unsigned char* in,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(2)
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    _mm512_mask_storeu_epi8(out + i, mask, in0);
  }
#else
  ref::mov_ker(out, in, len);
#endif
}

static inline __attribute__((always_inline)) void move_ker(
    bool* out,
    const bool* in,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(2)
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    _mm512_mask_storeu_epi8(out + i, mask, in0);
  }
#else
  ref::mov_ker(out, in, len);
#endif
}

static inline __attribute__((always_inline)) void move_ker(
    int8_t* out,
    const int8_t* in,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(2)
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    _mm512_mask_storeu_epi8(out + i, mask, in0);
  }
#else
  ref::mov_ker(out, in, len);
#endif
}

static inline void move_ker(int8_t* out, const int32_t* in, int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
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
#else
  ref::mov_ker(out, in, len);
#endif
}

#if defined(CPU_AVX512)
static inline void move_ker(int8_t* out, const __m512i* in, int64_t len) {
  int64_t i;
#pragma unroll(2)
  for (i = 0; i < len; i++) {
    _mm512_storeu_si512(out + i * 64, in[i]);
  }
}
#endif

static inline void add_ker(int8_t* inout, int8_t* in, int64_t len) {
  /*
    for (int64_t i = 0; i < len; ++i) {
      inout[i] += in[i];
    }
  */
  int64_t i;
#if defined(CPU_AVX512)
#pragma unroll(2)
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    auto out = _mm512_loadu_si512(inout + i);
    out = _mm512_adds_epi8(out, in0); // add with saturate
    _mm512_storeu_si512(inout + i, out);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    auto out = _mm512_maskz_loadu_epi8(mask, inout + i);
    out = _mm512_adds_epi8(out, in0);
    _mm512_mask_storeu_epi8(inout + i, mask, out);
  }
#else
  ref::add_ker(inout, in, len);
#endif
}

#if defined(CPU_AVX512)
static inline __attribute__((always_inline)) void scale_and_store_int8_128(
    void* out,
    const void* in,
    __m512& scale) {
  auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
  auto in0_2_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 32)));
  auto in0_3_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 48)));
  auto in0_4_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 64)));
  auto in0_5_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 80)));
  auto in0_6_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 96)));
  auto in0_7_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 112)));
  auto in0_0_32f = _mm512_cvt_roundepi32_ps(
      in0_0_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_1_32f = _mm512_cvt_roundepi32_ps(
      in0_1_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_2_32f = _mm512_cvt_roundepi32_ps(
      in0_2_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_3_32f = _mm512_cvt_roundepi32_ps(
      in0_3_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_4_32f = _mm512_cvt_roundepi32_ps(
      in0_4_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_5_32f = _mm512_cvt_roundepi32_ps(
      in0_5_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_6_32f = _mm512_cvt_roundepi32_ps(
      in0_6_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_7_32f = _mm512_cvt_roundepi32_ps(
      in0_7_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_0_32f = _mm512_mul_round_ps(
      in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_1_32f = _mm512_mul_round_ps(
      in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_2_32f = _mm512_mul_round_ps(
      in0_2_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_3_32f = _mm512_mul_round_ps(
      in0_3_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_4_32f = _mm512_mul_round_ps(
      in0_4_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_5_32f = _mm512_mul_round_ps(
      in0_5_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_6_32f = _mm512_mul_round_ps(
      in0_6_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_7_32f = _mm512_mul_round_ps(
      in0_7_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_0_32i = _mm512_cvt_roundps_epi32(
      in0_0_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_1_32i = _mm512_cvt_roundps_epi32(
      in0_1_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_2_32i = _mm512_cvt_roundps_epi32(
      in0_2_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_3_32i = _mm512_cvt_roundps_epi32(
      in0_3_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_4_32i = _mm512_cvt_roundps_epi32(
      in0_4_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_5_32i = _mm512_cvt_roundps_epi32(
      in0_5_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_6_32i = _mm512_cvt_roundps_epi32(
      in0_6_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_7_32i = _mm512_cvt_roundps_epi32(
      in0_7_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
  _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
  _mm_storeu_si128((__m128i*)(out + 32), _mm512_cvtsepi32_epi8(in0_2_32i));
  _mm_storeu_si128((__m128i*)(out + 48), _mm512_cvtsepi32_epi8(in0_3_32i));
  _mm_storeu_si128((__m128i*)(out + 64), _mm512_cvtsepi32_epi8(in0_4_32i));
  _mm_storeu_si128((__m128i*)(out + 80), _mm512_cvtsepi32_epi8(in0_5_32i));
  _mm_storeu_si128((__m128i*)(out + 96), _mm512_cvtsepi32_epi8(in0_6_32i));
  _mm_storeu_si128((__m128i*)(out + 112), _mm512_cvtsepi32_epi8(in0_7_32i));
}

static inline void scale_and_store_int8_64(
    void* out,
    const void* in,
    __m512& scale) {
  auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
  auto in0_2_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 32)));
  auto in0_3_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 48)));
  auto in0_0_32f = _mm512_cvt_roundepi32_ps(
      in0_0_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_1_32f = _mm512_cvt_roundepi32_ps(
      in0_1_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_2_32f = _mm512_cvt_roundepi32_ps(
      in0_2_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_3_32f = _mm512_cvt_roundepi32_ps(
      in0_3_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_0_32f = _mm512_mul_round_ps(
      in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_1_32f = _mm512_mul_round_ps(
      in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_2_32f = _mm512_mul_round_ps(
      in0_2_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_3_32f = _mm512_mul_round_ps(
      in0_3_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_0_32i = _mm512_cvt_roundps_epi32(
      in0_0_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_1_32i = _mm512_cvt_roundps_epi32(
      in0_1_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_2_32i = _mm512_cvt_roundps_epi32(
      in0_2_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_3_32i = _mm512_cvt_roundps_epi32(
      in0_3_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
  _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
  _mm_storeu_si128((__m128i*)(out + 32), _mm512_cvtsepi32_epi8(in0_2_32i));
  _mm_storeu_si128((__m128i*)(out + 48), _mm512_cvtsepi32_epi8(in0_3_32i));
}

static inline void scale_and_store_int8_32(
    void* out,
    const void* in,
    __m512& scale) {
  auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
  auto in0_0_32f = _mm512_cvt_roundepi32_ps(
      in0_0_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto in0_1_32f = _mm512_cvt_roundepi32_ps(
      in0_1_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_0_32f = _mm512_mul_round_ps(
      in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_1_32f = _mm512_mul_round_ps(
      in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_0_32i = _mm512_cvt_roundps_epi32(
      in0_0_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_1_32i = _mm512_cvt_roundps_epi32(
      in0_1_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
  _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
}

static inline void scale_and_store_int8_16(
    void* out,
    const void* in,
    __m512& scale) {
  auto in0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
  auto in0_32f = _mm512_cvt_roundepi32_ps(
      in0_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_32f = _mm512_mul_round_ps(
      in0_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_32i = _mm512_cvt_roundps_epi32(
      in0_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_32i));
}

static inline void scale_and_store_int8_maskz_16(
    void* out,
    const void* in,
    __m512& scale,
    __mmask8 mask) {
  auto in0 = _mm_maskz_loadu_epi8(mask, in);
  auto in0_32i = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(mask, in));
  auto in0_32f = _mm512_cvt_roundepi32_ps(
      in0_32i, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_32f = _mm512_mul_round_ps(
      in0_32f, scale, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  in0_32i = _mm512_cvt_roundps_epi32(
      in0_32f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm_mask_storeu_epi8(out, mask, _mm512_cvtsepi32_epi8(in0_32i));
}

static inline __attribute__((always_inline)) void scale_and_move_ker_128(
    int8_t* out,
    const int8_t* in,
    float scale) {
  if (std::abs(scale - 1.0) < 0.0005) {
    move_ker(out, in, 128);
  } else {
    __m512 scale_vec512 = _mm512_set1_ps(scale);
    scale_and_store_int8_128((void*)out, (const void*)in, scale_vec512);
  }
}
#else
static inline __attribute__((always_inline)) void scale_and_store_int8(
    int8_t* out,
    const int8_t* in,
    float& scale,
    int64_t len) {
  for (int64_t i = 0; i < len; i++) {
    int32_t i32_val = *(in + i);
    float ps_val = (float)i32_val;
    ps_val *= scale;
    i32_val = int32_t(std::round(ps_val));
    if (i32_val < INT8_MIN) {
      *(out + i) = INT8_MIN;
    } else if (i32_val > INT8_MAX) {
      *(out + i) = INT8_MAX;
    } else {
      *(out + i) = (int8_t)i32_val;
    }
  }
}
#endif

static inline void scale_and_move_ker(
    int8_t* out,
    const int8_t* in,
    float scale,
    int64_t len) {
  int64_t i;
#if defined(CPU_AVX512)
  __m512 scale_vec512 = _mm512_set1_ps(scale);
  for (i = 0; i < len - 127; i += 128) {
    scale_and_store_int8_128(
        (void*)(out + i), (const void*)(in + i), scale_vec512);
  }
  if ((len - i) > 63) {
    scale_and_store_int8_64(
        (void*)(out + i), (const void*)(in + i), scale_vec512);
    i += 64;
  }
  if ((len - i) > 31) {
    scale_and_store_int8_32(
        (void*)(out + i), (const void*)(in + i), scale_vec512);
    i += 32;
  }
  if ((len - i) > 15) {
    scale_and_store_int8_16(
        (void*)(out + i), (const void*)(in + i), scale_vec512);
    i += 16;
  }
  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    scale_and_store_int8_maskz_16(out + i, in + i, scale_vec512, mask);
  }
#else
  scale_and_store_int8(out, in, scale, len);
#endif
}

#if defined(CPU_AVX512)
static inline __attribute__((always_inline)) void mul_and_sum_s8x128_to_s32x16(
    __m512i& out,
    const int8_t* a,
    const int8_t* b) {
  auto a_0 = _mm256_loadu_si256((__m256i*)a);
  auto a_1 = _mm256_loadu_si256((__m256i*)(a + 32));
  auto a_2 = _mm256_loadu_si256((__m256i*)(a + 64));
  auto a_3 = _mm256_loadu_si256((__m256i*)(a + 96));
  auto b_0 = _mm256_loadu_si256((__m256i*)b);
  auto b_1 = _mm256_loadu_si256((__m256i*)(b + 32));
  auto b_2 = _mm256_loadu_si256((__m256i*)(b + 64));
  auto b_3 = _mm256_loadu_si256((__m256i*)(b + 96));
  auto a_0_i = _mm512_cvtepi8_epi16(a_0);
  auto a_1_i = _mm512_cvtepi8_epi16(a_1);
  auto a_2_i = _mm512_cvtepi8_epi16(a_2);
  auto a_3_i = _mm512_cvtepi8_epi16(a_3);
  auto b_0_i = _mm512_cvtepi8_epi16(b_0);
  auto b_1_i = _mm512_cvtepi8_epi16(b_1);
  auto b_2_i = _mm512_cvtepi8_epi16(b_2);
  auto b_3_i = _mm512_cvtepi8_epi16(b_3);
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
  out = _mm512_add_epi32(a_0_i, a_2_i);
}

static inline __attribute__((always_inline)) void load_s8x128_to_s16x128(
    __m512i* out_s16x4,
    const int8_t* in) {
  auto in_0 = _mm256_loadu_si256((__m256i*)in);
  auto in_1 = _mm256_loadu_si256((__m256i*)(in + 32));
  auto in_2 = _mm256_loadu_si256((__m256i*)(in + 64));
  auto in_3 = _mm256_loadu_si256((__m256i*)(in + 96));
  out_s16x4[0] = _mm512_cvtepi8_epi16(in_0);
  out_s16x4[1] = _mm512_cvtepi8_epi16(in_1);
  out_s16x4[2] = _mm512_cvtepi8_epi16(in_2);
  out_s16x4[3] = _mm512_cvtepi8_epi16(in_3);
}

static inline __attribute__((always_inline)) void load_s8x128x2_to_s16x128x2(
    __m512i* out_s16x8,
    const int8_t* in0,
    const int8_t* in1) {
  auto in0_0 = _mm256_loadu_si256((__m256i*)in0);
  auto in0_1 = _mm256_loadu_si256((__m256i*)(in0 + 32));
  auto in0_2 = _mm256_loadu_si256((__m256i*)(in0 + 64));
  auto in0_3 = _mm256_loadu_si256((__m256i*)(in0 + 96));
  auto in1_0 = _mm256_loadu_si256((__m256i*)in1);
  auto in1_1 = _mm256_loadu_si256((__m256i*)(in1 + 32));
  auto in1_2 = _mm256_loadu_si256((__m256i*)(in1 + 64));
  auto in1_3 = _mm256_loadu_si256((__m256i*)(in1 + 96));
  out_s16x8[0] = _mm512_cvtepi8_epi16(in0_0);
  out_s16x8[1] = _mm512_cvtepi8_epi16(in0_1);
  out_s16x8[2] = _mm512_cvtepi8_epi16(in0_2);
  out_s16x8[3] = _mm512_cvtepi8_epi16(in0_3);
  out_s16x8[4] = _mm512_cvtepi8_epi16(in1_0);
  out_s16x8[5] = _mm512_cvtepi8_epi16(in1_1);
  out_s16x8[6] = _mm512_cvtepi8_epi16(in1_2);
  out_s16x8[7] = _mm512_cvtepi8_epi16(in1_3);
}

static inline __attribute__((always_inline)) void mul_and_sum_s16x128_to_s32x16(
    __m512i& out,
    const __m512i* a16x4,
    const __m512i* b16x4) {
  auto a_0_i = _mm512_madd_epi16(a16x4[0], b16x4[0]);
  auto a_2_i = _mm512_madd_epi16(a16x4[2], b16x4[2]);
#ifdef AVX512_VNNI
  a_0_i = _mm512_dpwssd_epi32(a_0_i, a16x4[1], b16x4[1]);
  a_2_i = _mm512_dpwssd_epi32(a_2_i, a16x4[3], b16x4[3]);
#else
  auto a_1_i = _mm512_madd_epi16(a16x4[1], b16x4[1]);
  auto a_3_i = _mm512_madd_epi16(a16x4[3], b16x4[3]);
  a_0_i = _mm512_add_epi32(a_0_i, a_1_i);
  a_2_i = _mm512_add_epi32(a_2_i, a_3_i);
#endif
  out = _mm512_add_epi32(a_0_i, a_2_i);
}

static inline __attribute__((always_inline)) void
mul_and_sum_s8x128x2_to_s32x16x2(
    __m512i& out0,
    __m512i& out1,
    const int8_t* a0,
    const int8_t* b0,
    const int8_t* a1,
    const int8_t* b1) {
  auto a0_0 = _mm256_loadu_si256((__m256i*)a0);
  auto a0_1 = _mm256_loadu_si256((__m256i*)(a0 + 32));
  auto a0_2 = _mm256_loadu_si256((__m256i*)(a0 + 64));
  auto a0_3 = _mm256_loadu_si256((__m256i*)(a0 + 96));
  auto b0_0 = _mm256_loadu_si256((__m256i*)b0);
  auto b0_1 = _mm256_loadu_si256((__m256i*)(b0 + 32));
  auto b0_2 = _mm256_loadu_si256((__m256i*)(b0 + 64));
  auto b0_3 = _mm256_loadu_si256((__m256i*)(b0 + 96));
  auto a0_0_i = _mm512_cvtepi8_epi16(a0_0);
  auto a0_1_i = _mm512_cvtepi8_epi16(a0_1);
  auto a0_2_i = _mm512_cvtepi8_epi16(a0_2);
  auto a0_3_i = _mm512_cvtepi8_epi16(a0_3);
  auto b0_0_i = _mm512_cvtepi8_epi16(b0_0);
  auto b0_1_i = _mm512_cvtepi8_epi16(b0_1);
  auto b0_2_i = _mm512_cvtepi8_epi16(b0_2);
  auto b0_3_i = _mm512_cvtepi8_epi16(b0_3);
  auto a1_0 = _mm256_loadu_si256((__m256i*)a1);
  auto a1_1 = _mm256_loadu_si256((__m256i*)(a1 + 32));
  auto a1_2 = _mm256_loadu_si256((__m256i*)(a1 + 64));
  auto a1_3 = _mm256_loadu_si256((__m256i*)(a1 + 96));
  auto b1_0 = _mm256_loadu_si256((__m256i*)b1);
  auto b1_1 = _mm256_loadu_si256((__m256i*)(b1 + 32));
  auto b1_2 = _mm256_loadu_si256((__m256i*)(b1 + 64));
  auto b1_3 = _mm256_loadu_si256((__m256i*)(b1 + 96));
  auto a1_0_i = _mm512_cvtepi8_epi16(a1_0);
  auto a1_1_i = _mm512_cvtepi8_epi16(a1_1);
  auto a1_2_i = _mm512_cvtepi8_epi16(a1_2);
  auto a1_3_i = _mm512_cvtepi8_epi16(a1_3);
  auto b1_0_i = _mm512_cvtepi8_epi16(b1_0);
  auto b1_1_i = _mm512_cvtepi8_epi16(b1_1);
  auto b1_2_i = _mm512_cvtepi8_epi16(b1_2);
  auto b1_3_i = _mm512_cvtepi8_epi16(b1_3);
  a0_0_i = _mm512_madd_epi16(a0_0_i, b0_0_i);
  a0_2_i = _mm512_madd_epi16(a0_2_i, b0_2_i);
  a1_0_i = _mm512_madd_epi16(a1_0_i, b1_0_i);
  a1_2_i = _mm512_madd_epi16(a1_2_i, b1_2_i);
#ifdef AVX512_VNNI
  a0_0_i = _mm512_dpwssd_epi32(a0_0_i, a0_1_i, b0_1_i);
  a1_0_i = _mm512_dpwssd_epi32(a1_0_i, a1_1_i, b1_1_i);
  a0_2_i = _mm512_dpwssd_epi32(a0_2_i, a0_3_i, b0_3_i);
  a1_2_i = _mm512_dpwssd_epi32(a1_2_i, a1_3_i, b1_3_i);
#else
  a0_1_i = _mm512_madd_epi16(a0_1_i, b0_1_i);
  a0_3_i = _mm512_madd_epi16(a0_3_i, b0_3_i);
  a1_1_i = _mm512_madd_epi16(a1_1_i, b1_1_i);
  a1_3_i = _mm512_madd_epi16(a1_3_i, b1_3_i);
  a0_0_i = _mm512_add_epi32(a0_0_i, a0_1_i);
  a0_2_i = _mm512_add_epi32(a0_2_i, a0_3_i);
  a1_0_i = _mm512_add_epi32(a1_0_i, a1_1_i);
  a1_2_i = _mm512_add_epi32(a1_2_i, a1_3_i);
#endif
  out0 = _mm512_add_epi32(a0_0_i, a0_2_i);
  out1 = _mm512_add_epi32(a1_0_i, a1_2_i);
}

static inline __attribute__((always_inline)) void
mul_and_sum_s16x128x2_to_s32x16x2(
    __m512i& out0,
    __m512i& out1,
    const __m512i* a0_16x4,
    const __m512i* b0_16x4,
    const __m512i* a1_16x4,
    const __m512i* b1_16x4) {
  auto a0_0_i = _mm512_madd_epi16(a0_16x4[0], b0_16x4[0]);
  auto a1_0_i = _mm512_madd_epi16(a1_16x4[0], b1_16x4[0]);
  auto a0_2_i = _mm512_madd_epi16(a0_16x4[2], b0_16x4[2]);
  auto a1_2_i = _mm512_madd_epi16(a1_16x4[2], b1_16x4[2]);
#ifdef AVX512_VNNI
  a0_0_i = _mm512_dpwssd_epi32(a0_0_i, a0_16x4[1], b0_16x4[1]);
  a1_0_i = _mm512_dpwssd_epi32(a1_0_i, a1_16x4[1], b1_16x4[1]);
  a0_2_i = _mm512_dpwssd_epi32(a0_2_i, a0_16x4[3], b0_16x4[3]);
  a1_2_i = _mm512_dpwssd_epi32(a1_2_i, a1_16x4[3], b1_16x4[3]);
#else
  auto a0_1_i = _mm512_madd_epi16(a0_16x4[1], b0_16x4[1]);
  auto a0_3_i = _mm512_madd_epi16(a0_16x4[3], b0_16x4[3]);
  auto a1_1_i = _mm512_madd_epi16(a1_16x4[1], b1_16x4[1]);
  auto a1_3_i = _mm512_madd_epi16(a1_16x4[3], b1_16x4[3]);
  a0_0_i = _mm512_add_epi32(a0_0_i, a0_1_i);
  a0_2_i = _mm512_add_epi32(a0_2_i, a0_3_i);
  a1_0_i = _mm512_add_epi32(a1_0_i, a1_1_i);
  a1_2_i = _mm512_add_epi32(a1_2_i, a1_3_i);
#endif
  out0 = _mm512_add_epi32(a0_0_i, a0_2_i);
  out1 = _mm512_add_epi32(a1_0_i, a1_2_i);
}

static inline int32_t reduce_add_s32x16(__m512i& acc_sum) {
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

#if 0
static inline int8_t reduce_add_s32x16_with_scale(__m512i& acc_sum, float& scale) {
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
#endif

static inline __attribute__((always_inline)) __m512i reduce_add_s32x16x16(
    __m512i* acc_sums) {
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
  l1 = _mm512_unpacklo_epi64(l0, l2);
  l3 = _mm512_unpackhi_epi64(l0, l2);
  l5 = _mm512_unpacklo_epi64(l4, l6);
  l7 = _mm512_unpackhi_epi64(l4, l6);
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
  h1 = _mm512_unpacklo_epi64(h0, h2);
  h3 = _mm512_unpackhi_epi64(h0, h2);
  h5 = _mm512_unpacklo_epi64(h4, h6);
  h7 = _mm512_unpackhi_epi64(h4, h6);
  h1 = _mm512_add_epi32(h1, h3);
  h5 = _mm512_add_epi32(h5, h7);
  h0 = _mm512_shuffle_i32x4(h1, h5, 0x88);
  h2 = _mm512_shuffle_i32x4(h1, h5, 0xdd);
  h0 = _mm512_add_epi32(h0, h2);

  l1 = _mm512_shuffle_i32x4(l0, h0, 0x88);
  h1 = _mm512_shuffle_i32x4(l0, h0, 0xdd);
  l1 = _mm512_add_epi32(l1, h1);

  return l1;
}

static inline __attribute__((always_inline)) void
reduce_add_s32x16x16_with_scales(
    int8_t* outs,
    __m512i* acc_sums,
    const __m512& scales) {
  auto l1 = reduce_add_s32x16x16(acc_sums);
  auto l1_f = _mm512_cvtepi32_ps(l1);
  l1_f = _mm512_mul_round_ps(
      l1_f, scales, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  l1 = _mm512_cvtps_epi32(l1_f);
  auto out_16 = _mm512_cvtsepi32_epi8(l1);
  _mm_storeu_si128((__m128i*)outs, out_16);
}

static inline __attribute__((always_inline)) void
reduce_add_s32x16x16x4_with_scales(
    int8_t* outs,
    __m512i* acc_sums,
    const __m512 (&scales)[4]) {
  auto l1 = reduce_add_s32x16x16(acc_sums);
  auto l2 = reduce_add_s32x16x16(acc_sums + 16);
  auto l3 = reduce_add_s32x16x16(acc_sums + 32);
  auto l4 = reduce_add_s32x16x16(acc_sums + 48);
  auto l1_f = _mm512_cvtepi32_ps(l1);
  auto l2_f = _mm512_cvtepi32_ps(l2);
  auto l3_f = _mm512_cvtepi32_ps(l3);
  auto l4_f = _mm512_cvtepi32_ps(l4);
  l1_f = _mm512_mul_round_ps(
      l1_f, scales[0], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  l2_f = _mm512_mul_round_ps(
      l2_f, scales[1], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  l3_f = _mm512_mul_round_ps(
      l3_f, scales[2], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  l4_f = _mm512_mul_round_ps(
      l4_f, scales[3], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  l1 = _mm512_cvtps_epi32(l1_f);
  l2 = _mm512_cvtps_epi32(l2_f);
  l3 = _mm512_cvtps_epi32(l3_f);
  l4 = _mm512_cvtps_epi32(l4_f);
  auto out1_16 = _mm512_cvtsepi32_epi8(l1);
  auto out2_16 = _mm512_cvtsepi32_epi8(l2);
  auto out3_16 = _mm512_cvtsepi32_epi8(l3);
  auto out4_16 = _mm512_cvtsepi32_epi8(l4);
  _mm_storeu_si128((__m128i*)outs, out1_16);
  _mm_storeu_si128((__m128i*)(outs + 16), out2_16);
  _mm_storeu_si128((__m128i*)(outs + 32), out3_16);
  _mm_storeu_si128((__m128i*)(outs + 48), out4_16);
}

static inline __attribute__((always_inline)) void
reduce_add_s32x16x16_with_scales_and_mask_store(
    int8_t* outs,
    __mmask16 mask,
    __m512i* acc_sums,
    const __m512& scales) {
  auto l1 = reduce_add_s32x16x16(acc_sums);
  auto l1_f = _mm512_cvtepi32_ps(l1);
  l1_f = _mm512_mul_round_ps(
      l1_f, scales, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  l1 = _mm512_cvtps_epi32(l1_f);
  auto out_16 = _mm512_cvtsepi32_epi8(l1);
  _mm_mask_storeu_epi8((__m128i*)outs, mask, out_16);
}

static inline __attribute__((always_inline)) int32_t mul_and_sum_int8_128(
    const int8_t* a,
    const int8_t* b) {
  __m512i acc_sum;
  mul_and_sum_s8x128_to_s32x16(acc_sum, a, b);
  return reduce_add_s32x16(acc_sum);
}

static inline __attribute__((always_inline)) int32_t mul_and_sum_int8_64(
    const int8_t* a,
    const int8_t* b) {
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
#endif

#if defined(CPU_AVX512)
static inline __attribute__((always_inline)) int32_t _scale_int32(
    int32_t value,
    float scale) {
  auto v_simd = _mm_setzero_ps();
  auto s_simd = _mm_set1_ps(scale);
  v_simd = _mm_cvt_si2ss(v_simd, value);
  v_simd = _mm_mul_round_ss(
      v_simd, s_simd, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  int32_t c = _mm_cvt_roundss_si32(
      v_simd, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto c_simd = _mm_set1_epi32(c);
  c_simd = _mm_cvtsepi32_epi8(c_simd);
  c = _mm_cvtsi128_si32(c_simd);
  return c;
}
#else
static inline __attribute__((always_inline)) int32_t _scale_int32(
    int32_t value,
    float scale) {
  float f_val = float(value) * scale;
  int32_t i32_val = int32_t(std::round(f_val));
  if (i32_val < INT8_MIN) {
    i32_val = INT8_MIN;
  } else if (i32_val > INT8_MAX) {
    i32_val = INT8_MAX;
  }
  return i32_val;
}

#endif
static inline __attribute__((always_inline)) int8_t _dot_s8s8_scale_s32s8(
    const int8_t* a,
    const int8_t* b,
    size_t len,
    float scale) {
  int32_t c = 0;
  size_t i = 0;
#if defined(CPU_AVX512)
  for (; i < len - 127; i += 128) {
    c += mul_and_sum_int8_128(a + i, b + i);
  }
  if ((len - i) > 63) {
    c += mul_and_sum_int8_64(a + i, b + i);
    i += 64;
  }
#endif
  for (; i < len; i++) {
    c += (int32_t)a[i] * (int32_t)b[i];
  }
  c = _scale_int32(c, scale);
  return (int8_t)c;
}
