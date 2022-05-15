#pragma once
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec512/vec512.h>
using namespace at::vec;

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
} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at

#include <immintrin.h>

namespace torch_ipex {
namespace cpu {
namespace kernel {

inline __m512 pack_bf16_to_fp32(const __m256i top, const __m256i bot) {
  auto x1 = _mm512_cvtepu16_epi32(top);
  auto x2 = _mm512_cvtepu16_epi32(bot);
  auto y = _mm512_add_epi32(_mm512_bslli_epi128(x1, 2), x2);
  return _mm512_castsi512_ps(y);
}

// Only support AVX512 impl at current stage. Will expand this impl to cover
// AVX2 and other cases.
inline void packed_bf16_add_ker(
    at::BFloat16* a1,
    at::BFloat16* a2,
    at::BFloat16* b,
    int len,
    float alpha) {
  auto vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 31; i += 32) {
    auto x10 = _mm256_loadu_si256((__m256i*)(a1 + i));
    auto x11 = _mm256_loadu_si256((__m256i*)(a1 + i + 16));
    auto x20 = _mm256_loadu_si256((__m256i*)(a2 + i));
    auto x21 = _mm256_loadu_si256((__m256i*)(a2 + i + 16));
    auto y10 = _mm256_loadu_si256((__m256i*)(b + i));
    auto y11 = _mm256_loadu_si256((__m256i*)(b + i + 16));

    auto z10 = pack_bf16_to_fp32(x10, x20);
    auto z20 = cvt_bf16_to_fp32(y10);
    z10 = _mm512_fmadd_ps(vAlpha, z20, z10);
    auto z11 = pack_bf16_to_fp32(x11, x21);
    auto z21 = cvt_bf16_to_fp32(y11);
    z11 = _mm512_fmadd_ps(vAlpha, z21, z11);
    // Update result back to split input tensors.
    _mm256_storeu_si256((__m256i*)(a1 + i), trunc_fp32_to_bf16(z10));
    _mm256_storeu_si256(
        (__m256i*)(a2 + i), _mm512_cvtepi32_epi16(_mm512_castps_si512(z10)));
    _mm256_storeu_si256((__m256i*)(a1 + i + 16), trunc_fp32_to_bf16(z11));
    _mm256_storeu_si256(
        (__m256i*)(a2 + i + 16),
        _mm512_cvtepi32_epi16(_mm512_castps_si512(z11)));
  }

  if (i < len) {
    for (; i < len - 15; i += 16) {
      auto x1 = _mm256_loadu_si256((__m256i*)(a1 + i));
      auto x2 = _mm256_loadu_si256((__m256i*)(a2 + i));
      auto y1 = _mm256_loadu_si256((__m256i*)(b + i));

      auto z1 = pack_bf16_to_fp32(x1, x2);
      auto z2 = cvt_bf16_to_fp32(y1);
      z1 = _mm512_fmadd_ps(vAlpha, z2, z1);
      // Update result back to split input tensors.
      _mm256_storeu_si256((__m256i*)(a1 + i), trunc_fp32_to_bf16(z1));
      _mm256_storeu_si256(
          (__m256i*)(a2 + i), _mm512_cvtepi32_epi16(_mm512_castps_si512(z1)));
    }

    __mmask16 mask = (1 << (len - i)) - 1;
    auto x1 = _mm256_maskz_loadu_epi16(mask, a1 + i);
    auto x2 = _mm256_maskz_loadu_epi16(mask, a2 + i);
    auto y1 = _mm256_maskz_loadu_epi16(mask, b + i);

    auto z1 = pack_bf16_to_fp32(x1, x2);
    auto z2 = cvt_bf16_to_fp32(y1);
    z1 = _mm512_fmadd_ps(vAlpha, z2, z1);
    // Update result back to split input tensors.
    _mm256_mask_storeu_epi16(a1 + i, mask, trunc_fp32_to_bf16(z1));
    _mm256_mask_storeu_epi16(
        a2 + i, mask, _mm512_cvtepi32_epi16(_mm512_castps_si512(z1)));
  }
}

template <>
inline __attribute__((always_inline)) void add_ker(
    at::BFloat16* inout,
    at::BFloat16* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i)));
    auto inout2 =
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(inout + i + 16)));
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

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto inout1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_bf16(inout1));
  }
}

template <>
inline __attribute__((always_inline)) void add_ker(
    float* inout,
    float* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
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

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto out1 = _mm512_maskz_loadu_ps(mask, inout + i);
    auto in1 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(inout + i, mask, _mm512_add_ps(out1, in1));
  }
}

template <>
inline __attribute__((always_inline)) void add_ker(
    float* inout,
    at::BFloat16* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(2)
  for (i = 0; i < len - 31; i += 32) {
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

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    auto inout1 = _mm512_maskz_loadu_ps(mask, inout + i);
    inout1 = _mm512_add_ps(inout1, in1);
    _mm512_mask_storeu_ps(inout + i, mask, inout1);
  }
}

template <>
inline __attribute__((always_inline)) void move_ker(
    at::BFloat16* out,
    const float* in,
    int64_t len) {
  cvt_fp32_to_bf16(out, in, len);
}

static inline __attribute__((always_inline)) void move_ker_load_aligned(
    at::BFloat16* out,
    float* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = cvt_fp32_to_bf16(_mm512_load_ps(in + i));
    auto in1 = cvt_fp32_to_bf16(_mm512_load_ps(in + i + 16));
    _mm256_storeu_si256((__m256i*)(out + i), in0);
    _mm256_storeu_si256((__m256i*)(out + i + 16), in1);
  }

  if (i < len - 15) {
    auto in0 = cvt_fp32_to_bf16(_mm512_load_ps(in + i));
    _mm256_storeu_si256((__m256i*)(out + i), in0);
    i += 16;
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = cvt_fp32_to_bf16(_mm512_maskz_load_ps(mask, in + i));
    _mm256_mask_storeu_epi16((__m256i*)(out + i), mask, in0);
  }
}

template <>
inline __attribute__((always_inline)) void move_ker(
    float* out,
    const float* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
}

static inline __attribute__((always_inline)) void move_ker_load_aligned(
    float* out,
    const float* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_load_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_load_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
}

template <>
inline __attribute__((always_inline)) void move_ker(
    at::BFloat16* out,
    const at::BFloat16* in,
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

static inline __attribute__((always_inline)) void move_ker_load_aligned(
    at::BFloat16* out,
    const at::BFloat16* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = _mm512_load_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in0 = _mm512_maskz_loadu_epi16(mask, in + i);
    _mm512_mask_storeu_epi16(out + i, mask, in0);
  }
}

template <>
inline __attribute__((always_inline)) void move_ker(
    int64_t* out,
    int64_t* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 7; i += 8) {
    auto in0 = _mm512_loadu_pd(in + i);
    _mm512_storeu_pd(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_pd(mask, in + i);
    _mm512_mask_storeu_pd(out + i, mask, in0);
  }
}

template <>
inline __attribute__((always_inline)) void move_ker(
    int32_t* out,
    const int32_t* in,
    int64_t len) {
  int64_t i = 0;
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
}

static inline __attribute__((always_inline)) void zero_ker(
    float* out,
    int64_t len) {
  int64_t i = 0;
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

static inline __attribute__((always_inline)) void zero_ker(
    at::BFloat16* out,
    int64_t len) {
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

inline __m512 convert_bf16_to_fp32(const __m256i src) {
  __m512i y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
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

template <>
inline void madd_ker(float* inout, at::BFloat16* in, int len, float alpha) {
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    __m512 y1 = _mm512_loadu_ps(inout + i);
    __m512 y2 = convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in + i)));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_storeu_ps(inout + i, y1);
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = _mm512_maskz_loadu_ps(mask, inout + i);
    __m512 y2 = convert_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_mask_storeu_ps(inout + i, mask, y1);
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
