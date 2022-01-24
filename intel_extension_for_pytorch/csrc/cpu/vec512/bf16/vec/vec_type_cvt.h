#include <ATen/ATen.h>

#if defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/vec512.h>
#else
#include <ATen/cpu/vec/vec256/vec256.h>
#endif
using namespace at::vec;

#if defined(CPU_CAPABILITY_AVX512)
#include <immintrin.h>
// Conversion from BF16 to FP32
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline void cvt_bf16_to_fp32(float* dst, const at::BFloat16* src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(src + i)));
    _mm512_storeu_ps(dst + i, f32);
  }
  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto f32 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, src + i));
    _mm512_mask_storeu_ps(dst + i, mask, f32);
  }
}

// Conversion from FP32 to BF16
inline __m256i trunc_fp32_to_bf16(const __m512 src) {
  auto y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if (defined CPU_CAPABILITY_AVX512_BF16)
  return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
#else
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
#endif
}

inline void cvt_fp32_to_bf16(at::BFloat16* dst, const float* src, int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256((__m256i*)(dst + i), cvt_fp32_to_bf16(f32));
  }
  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto f32 = _mm512_maskz_loadu_ps(mask, src + i);
    _mm256_mask_storeu_epi16(dst + i, mask, cvt_fp32_to_bf16(f32));
  }
}

#else // Not AVX512

inline void cvt_bf16_to_fp32(float* dst, const at::BFloat16* src, int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}

inline void cvt_fp32_to_bf16(at::BFloat16* dst, const float* src, int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}
#endif

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

#if defined(CPU_CAPABILITY_AVX512)
inline std::tuple<Vectorized<float>, Vectorized<float>> pack_bfloat16_float(
    const Vectorized<at::BFloat16>& a,
    const Vectorized<at::BFloat16>& b) {
  __m512i a0 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(__m512i(a), 0));
  __m512i a1 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(__m512i(a), 1));
  __m512i b0 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(__m512i(b), 0));
  __m512i b1 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(__m512i(b), 1));
  __m512 y0 =
      _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(a0, 16), b0));
  __m512 y1 =
      _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(a1, 16), b1));
  return std::make_tuple(y0, y1);
}

inline std::tuple<Vectorized<at::BFloat16>, Vectorized<at::BFloat16>>
unpack_float_bfloat16(const Vectorized<float>& a, const Vectorized<float>& b) {
  __m512i x0 = _mm512_castps_si512(__m512(a));
  __m512i x1 = _mm512_castps_si512(__m512(b));
  __m512i x0_hi = _mm512_srli_epi32(x0, 16);
  __m512i x1_hi = _mm512_srli_epi32(x1, 16);

  __m512i zeros = _mm512_set1_epi32(0xffff);
  __m512i x0_lo = _mm512_and_si512(x0, zeros);
  __m512i x1_lo = _mm512_and_si512(x1, zeros);

  __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  __m512i y0 = _mm512_packus_epi32(x0_hi, x1_hi);
  y0 = _mm512_permutexvar_epi64(idx, y0);
  __m512i y1 = _mm512_packus_epi32(x0_lo, x1_lo);
  y1 = _mm512_permutexvar_epi64(idx, y1);
  return std::make_tuple(y0, y1);
}
#else
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
#endif
} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
