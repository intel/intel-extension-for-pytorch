#pragma once
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec512/vec512.h>

#include "utils/SysUtil.h"

using namespace at::vec;

#include <immintrin.h>

// Conversion from FP16 to FP32
inline __m512 cvt_fp16_to_fp32(const __m256i src) {
  return _mm512_cvtph_ps(src);
}

// Conversion from FP32 to FP16
inline __m256i cvt_fp32_to_fp16(const __m512 src) {
  return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline void cvt_fp32_to_fp16(at::Half* dst, const float* src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256((__m256i*)(dst + i), cvt_fp32_to_fp16(f32));
  }
  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto f32 = _mm512_maskz_loadu_ps(mask, src + i);
    _mm256_mask_storeu_epi16(dst + i, mask, cvt_fp32_to_fp16(f32));
  }
}

namespace torch_ipex {
namespace cpu {
namespace kernel {

template <>
IPEX_FORCE_INLINE void move_ker(at::Half* out, const float* in, int64_t len) {
  cvt_fp32_to_fp16(out, in, len);
}

template <>
IPEX_FORCE_INLINE void move_ker(
    at::Half* out,
    const at::Half* in,
    int64_t len) {
  int64_t i = 0;
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

static IPEX_FORCE_INLINE void zero_ker(at::Half* out, int64_t len) {
  int64_t i = 0;
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

template <>
IPEX_FORCE_INLINE void add_ker(
    at::Half* inout,
    const at::Half* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto inout1 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i)));
    auto inout2 =
        cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i + 16)));
    auto in1 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    auto in2 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i + 16)));
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm256_storeu_si256((__m256i*)(inout + i), cvt_fp32_to_fp16(inout1));
    _mm256_storeu_si256((__m256i*)(inout + i + 16), cvt_fp32_to_fp16(inout2));
  }

  if (i < len - 15) {
    auto inout1 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i)));
    auto in1 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_storeu_si256((__m256i*)(inout + i), cvt_fp32_to_fp16(inout1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto inout1 = cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    auto in1 = cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_fp16(inout1));
  }
}

template <>
IPEX_FORCE_INLINE void add_ker(float* inout, const at::Half* in, int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto in1 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    auto in2 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i + 16)));
    auto inout1 = _mm512_loadu_ps(inout + i);
    auto inout2 = _mm512_loadu_ps(inout + i + 16);
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm512_storeu_ps(inout + i, inout1);
    _mm512_storeu_ps(inout + i + 16, inout2);
  }

  if (i < len - 15) {
    auto in1 = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    auto inout1 = _mm512_loadu_ps(inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_storeu_ps(inout + i, inout1);
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in1 = cvt_fp16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    auto inout1 = _mm512_maskz_loadu_ps(mask, inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_mask_storeu_ps(inout + i, mask, inout1);
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
