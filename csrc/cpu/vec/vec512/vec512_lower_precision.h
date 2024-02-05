#pragma once
#include "vec512_bfloat16.h"
#include "vec512_half.h"

// Conversion from BF16/FP16 to FP32
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m512 cvt_to_fp32(const __m256i src);

template <>
inline __m512 cvt_to_fp32<at::BFloat16>(const __m256i src) {
  return cvt_bf16_to_fp32(src);
}
template <>
inline __m512 cvt_to_fp32<at::Half>(const __m256i src) {
  return cvt_fp16_to_fp32(src);
}

// Conversion from FP32 to BF16/FP16
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m256i cvt_from_fp32(const __m512 src);

template <>
inline __m256i cvt_from_fp32<at::BFloat16>(const __m512 src) {
  return cvt_fp32_to_bf16(src);
}
template <>
inline __m256i cvt_from_fp32<at::Half>(const __m512 src) {
  return cvt_fp32_to_fp16(src);
}
