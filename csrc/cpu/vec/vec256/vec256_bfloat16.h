#pragma once
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec256/vec256.h>

#include "utils/SysUtil.h"

using namespace at::vec;

inline void cvt_bf16_to_fp32(float* dst, const at::BFloat16* src, int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}

inline void cvt_fp16_to_fp32(float* dst, const at::Half* src, int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}

inline void cvt_fp32_to_bf16(at::BFloat16* dst, const float* src, int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}

inline void cvt_fp32_to_fp16(at::Half* dst, const float* src, int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}

/*
  Following the namespace convention of PyTorch, we put ISA-specific kernels
  under at::vec::[CPU_CAPABILITY] with [CPU_CAPABILITY] as the inline namespace.
  Then, the signatures of kernel functions are declared and invoked the same way
  regardless of ISAs. See Note [CPU_CAPABILITY namespace] in PyTorch.
 */
namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

inline float pack_bfloat16_float(at::BFloat16 a, at::BFloat16 b) {
  uint16_t* ap = reinterpret_cast<uint16_t*>(&a);
  uint16_t* bp = reinterpret_cast<uint16_t*>(&b);
  uint32_t hi = static_cast<uint32_t>(*ap);
  uint32_t lo = static_cast<uint32_t>(*bp);
  uint32_t out = (hi << 16) + lo;
  float* outp = reinterpret_cast<float*>(&out);
  return *outp;
}

inline std::tuple<at::BFloat16, at::BFloat16> unpack_float_bfloat16(float a) {
  uint32_t* ap = reinterpret_cast<uint32_t*>(&a);
  uint16_t hi = static_cast<uint16_t>((*ap) >> 16);
  uint16_t lo = static_cast<uint16_t>((*ap));
  at::BFloat16* hip = reinterpret_cast<at::BFloat16*>(&hi);
  at::BFloat16* lop = reinterpret_cast<at::BFloat16*>(&lo);
  return std::make_tuple(*hip, *lop);
}

inline std::tuple<Vectorized<float>, Vectorized<float>> pack_bfloat16_float(
    const Vectorized<at::BFloat16>& a,
    const Vectorized<at::BFloat16>& b) {
  __m256i a0 = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(__m256i(a), 0));
  __m256i a1 = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(__m256i(a), 1));
  __m256i b0 = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(__m256i(b), 0));
  __m256i b1 = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(__m256i(b), 1));
  __m256 y0 =
      _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(a0, 16), b0));
  __m256 y1 =
      _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(a1, 16), b1));
  return std::make_tuple(y0, y1);
}

inline std::tuple<Vectorized<at::BFloat16>, Vectorized<at::BFloat16>>
unpack_float_bfloat16(const Vectorized<float>& a, const Vectorized<float>& b) {
  __m256i x0 = _mm256_castps_si256(__m256(a));
  __m256i x1 = _mm256_castps_si256(__m256(b));
  __m256i x0_hi = _mm256_srli_epi32(x0, 16);
  __m256i x1_hi = _mm256_srli_epi32(x1, 16);

  __m256i zeros = _mm256_set1_epi32(0xffff);
  __m256i x0_lo = _mm256_and_si256(x0, zeros);
  __m256i x1_lo = _mm256_and_si256(x1, zeros);

  __m256i y0 = _mm256_packus_epi32(x0_hi, x1_hi);
  y0 = _mm256_permute4x64_epi64(y0, 0xd8);
  __m256i y1 = _mm256_packus_epi32(x0_lo, x1_lo);
  y1 = _mm256_permute4x64_epi64(y1, 0xd8);
  return std::make_tuple(y0, y1);
}
} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at

namespace torch_ipex {
namespace cpu {
namespace kernel {

inline void packed_bf16_add_ker(
    at::BFloat16* a1,
    at::BFloat16* a2,
    at::BFloat16* b,
    int len,
    float alpha) {
  for (int i = 0; i < len; i++) {
    uint32_t hi = (a1 + i)->x;
    uint32_t lo = (a2 + i)->x;
    uint32_t merge = hi << 16 | lo;
    float a_val = *((float*)&merge);
    float b_val = *(b + i);
    float res = a_val + b_val * alpha;
    (a1 + i)->x = (uint16_t)((*((uint32_t*)(&res))) >> 16);
    (a2 + i)->x = *((uint16_t*)(&res));
  }
}

static IPEX_FORCE_INLINE void move_ker_load_aligned(
    at::BFloat16* out,
    const float* in,
    int64_t len) {
  move_ker(out, in, len);
}

static IPEX_FORCE_INLINE void move_ker_load_aligned(
    float* out,
    const float* in,
    int64_t len) {
  move_ker(out, in, len);
}

static IPEX_FORCE_INLINE void move_ker_load_aligned(
    at::BFloat16* out,
    const at::BFloat16* in,
    int64_t len) {
  move_ker(out, in, len);
}

template <typename T>
inline float toFloat(T val) {
  float ret = float(val);
  return ret;
}

template <typename T1, typename T2>
inline void madd_ker(T1* inout, T2* in, int len, float alpha) {
#pragma omp simd
  for (long v = 0; v < len; v++) {
    inout[v] += toFloat(in[v]) * alpha;
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex