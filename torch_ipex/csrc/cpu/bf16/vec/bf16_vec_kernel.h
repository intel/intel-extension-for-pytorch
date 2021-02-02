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

inline void add_ker(at::BFloat16 *inout, at::BFloat16 *in, int len) {
  int i;
  #pragma unroll(2)
  for(i = 0; i < len - 31; i += 32) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i)));
    auto inout2 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i + 16)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    auto in2 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i + 16)));
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm256_storeu_si256((__m256i*)(inout + i), cvt_fp32_to_bf16(inout1));
    _mm256_storeu_si256((__m256i*)(inout + i + 16), cvt_fp32_to_bf16(inout2));
  }

  if (i < len - 15) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_storeu_si256((__m256i*)(inout + i), cvt_fp32_to_bf16(inout1));
    i += 16;
  }

  if(i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto inout1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_bf16(inout1));
  }
}

static inline void add_ker(float *inout, float *in, int len) {
  int i;
  #pragma unroll(2)
  for(i = 0; i < len - 31; i += 32) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto out2 = _mm512_loadu_ps(inout + i + 16);
    auto in1 = _mm512_loadu_ps(in + i);
    auto in2 = _mm512_loadu_ps(in + i + 16);
    out1 = _mm512_add_ps(out1, in1);
    out2 = _mm512_add_ps(out2, in2);
    _mm512_storeu_ps(inout + i, out1);
    _mm512_storeu_ps(inout + i + 16, out2);
  }

  if (i < len - 15) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto in1 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(inout + i, _mm512_add_ps(out1, in1));
    i += 16;
  }

  if(i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto out1 = _mm512_maskz_loadu_ps(mask, inout + i);
    auto in1 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(inout + i, mask, _mm512_add_ps(out1, in1));
  }
}

static inline void add_ker(float *inout, at::BFloat16 *in, int len) {
  int i;
  #pragma unroll(2)
  for(i = 0; i < len - 31; i += 32) {
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    auto in2 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i + 16)));
    auto inout1 = _mm512_loadu_ps(inout + i);
    auto inout2 = _mm512_loadu_ps(inout + i + 16);
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm512_storeu_ps(inout + i, inout1);
    _mm512_storeu_ps(inout + i + 16, inout2);
  }

  if (i < len - 15) {
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    auto inout1 = _mm512_loadu_ps(inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_storeu_ps(inout + i, inout1);
    i += 16;
  }

  if(i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    auto inout1 = _mm512_maskz_loadu_ps(mask, inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_mask_storeu_ps(inout + i, mask, inout1);
  }
}

static inline void move_ker(at::BFloat16 *out, float *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = cvt_fp32_to_bf16(_mm512_loadu_ps(in + i));
    auto in1 = cvt_fp32_to_bf16(_mm512_loadu_ps(in + i + 16));
    _mm256_storeu_si256((__m256i *)(out + i), in0);
    _mm256_storeu_si256((__m256i *)(out + i + 16), in1);
  }

  if (i < len - 15) {
    auto in0 = cvt_fp32_to_bf16(_mm512_loadu_ps(in + i));
    _mm256_storeu_si256((__m256i *)(out + i), in0);
    i += 16;
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = cvt_fp32_to_bf16(_mm512_maskz_loadu_ps(mask, in + i));
    _mm256_mask_storeu_epi16((__m256i *)(out + i), mask, in0);
  }
}

static inline void move_ker(float *out, const float *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 15 ; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i );
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
}

static inline void move_ker(at::BFloat16 *out, const at::BFloat16 *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in0 = _mm512_maskz_loadu_epi16(mask, in + i);
    _mm512_mask_storeu_epi16(out + i, mask, in0);
  }
}

static inline void move_ker(int64_t *out, int64_t *in, int64_t len) {
  int64_t i;
  #pragma unroll(4)
  for (i = 0; i < len - 7 ; i += 8) {
    auto in0 = _mm512_loadu_pd(in + i );
    _mm512_storeu_pd(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_pd(mask, in + i);
    _mm512_mask_storeu_pd(out + i, mask, in0);
  }
}

static inline void zero_ker(float *out, int64_t len) {
  int64_t i;
  __m512 zero_512 = _mm512_setzero_ps();
  #pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    _mm512_storeu_ps(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_ps(out + i, mask, zero_512);
  }
}

static inline void zero_ker(at::BFloat16 *out, int64_t len) {
  int64_t i;
  __m512i zero_512 = _mm512_setzero_si512();
  #pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi16(out + i, mask, zero_512);
  }
}
