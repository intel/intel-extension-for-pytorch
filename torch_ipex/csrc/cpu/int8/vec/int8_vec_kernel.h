#include <immintrin.h>
#include <cmath>

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

static inline void move_ker(int64_t *out, const int64_t *in, int64_t len) {
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

static inline void move_ker(int32_t *out, const int32_t *in, int64_t len) {
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

static inline void move_ker(int8_t *out, const int8_t *in, int64_t len) {
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
