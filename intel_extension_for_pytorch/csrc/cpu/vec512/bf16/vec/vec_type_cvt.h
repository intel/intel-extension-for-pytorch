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
  return trunc_fp32_to_bf16(src);
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
  // TODO: Vectorized version
  Vectorized<float> y0 = Vectorized<float>(
      pack_bfloat16_float(__m512i(a)[0], __m512i(b)[0]),
      pack_bfloat16_float(__m512i(a)[1], __m512i(b)[1]),
      pack_bfloat16_float(__m512i(a)[2], __m512i(b)[2]),
      pack_bfloat16_float(__m512i(a)[3], __m512i(b)[3]),
      pack_bfloat16_float(__m512i(a)[4], __m512i(b)[4]),
      pack_bfloat16_float(__m512i(a)[5], __m512i(b)[5]),
      pack_bfloat16_float(__m512i(a)[6], __m512i(b)[6]),
      pack_bfloat16_float(__m512i(a)[7], __m512i(b)[7]),
      pack_bfloat16_float(__m512i(a)[8], __m512i(b)[8]),
      pack_bfloat16_float(__m512i(a)[9], __m512i(b)[9]),
      pack_bfloat16_float(__m512i(a)[10], __m512i(b)[10]),
      pack_bfloat16_float(__m512i(a)[11], __m512i(b)[11]),
      pack_bfloat16_float(__m512i(a)[12], __m512i(b)[12]),
      pack_bfloat16_float(__m512i(a)[13], __m512i(b)[13]),
      pack_bfloat16_float(__m512i(a)[14], __m512i(b)[14]),
      pack_bfloat16_float(__m512i(a)[15], __m512i(b)[15]));
  Vectorized<float> y1 = Vectorized<float>(
      pack_bfloat16_float(__m512i(a)[16], __m512i(b)[16]),
      pack_bfloat16_float(__m512i(a)[17], __m512i(b)[17]),
      pack_bfloat16_float(__m512i(a)[18], __m512i(b)[18]),
      pack_bfloat16_float(__m512i(a)[19], __m512i(b)[19]),
      pack_bfloat16_float(__m512i(a)[20], __m512i(b)[20]),
      pack_bfloat16_float(__m512i(a)[21], __m512i(b)[21]),
      pack_bfloat16_float(__m512i(a)[22], __m512i(b)[22]),
      pack_bfloat16_float(__m512i(a)[23], __m512i(b)[23]),
      pack_bfloat16_float(__m512i(a)[24], __m512i(b)[24]),
      pack_bfloat16_float(__m512i(a)[25], __m512i(b)[25]),
      pack_bfloat16_float(__m512i(a)[26], __m512i(b)[26]),
      pack_bfloat16_float(__m512i(a)[27], __m512i(b)[27]),
      pack_bfloat16_float(__m512i(a)[28], __m512i(b)[28]),
      pack_bfloat16_float(__m512i(a)[29], __m512i(b)[29]),
      pack_bfloat16_float(__m512i(a)[30], __m512i(b)[30]),
      pack_bfloat16_float(__m512i(a)[31], __m512i(b)[31]));
  return std::make_tuple(y0, y1);
}

inline std::tuple<Vectorized<at::BFloat16>, Vectorized<at::BFloat16>>
unpack_float_bfloat16(const Vectorized<float>& a, const Vectorized<float>& b) {
  // TODO: Vectorized version
  std::vector<at::BFloat16> lo_val(32);
  std::vector<at::BFloat16> hi_val(32);
  for (int i = 0; i < 16; i++) {
    std::tie(lo_val[i], hi_val[i]) = unpack_float_bfloat16(__m512(a)[i]);
  }
  for (int i = 0; i < 16; i++) {
    std::tie(lo_val[i + 16], hi_val[i + 16]) =
        unpack_float_bfloat16(__m512(b)[i]);
  }
  Vectorized<at::BFloat16> y0 = Vectorized<at::BFloat16>(
      lo_val[0],
      lo_val[1],
      lo_val[2],
      lo_val[3],
      lo_val[4],
      lo_val[5],
      lo_val[6],
      lo_val[7],
      lo_val[8],
      lo_val[9],
      lo_val[10],
      lo_val[11],
      lo_val[12],
      lo_val[13],
      lo_val[14],
      lo_val[15],
      lo_val[16],
      lo_val[17],
      lo_val[18],
      lo_val[19],
      lo_val[20],
      lo_val[21],
      lo_val[22],
      lo_val[23],
      lo_val[24],
      lo_val[25],
      lo_val[26],
      lo_val[27],
      lo_val[28],
      lo_val[29],
      lo_val[30],
      lo_val[31]);
  Vectorized<at::BFloat16> y1 = Vectorized<at::BFloat16>(
      hi_val[0],
      hi_val[1],
      hi_val[2],
      hi_val[3],
      hi_val[4],
      hi_val[5],
      hi_val[6],
      hi_val[7],
      hi_val[8],
      hi_val[9],
      hi_val[10],
      hi_val[11],
      hi_val[12],
      hi_val[13],
      hi_val[14],
      hi_val[15],
      hi_val[16],
      hi_val[17],
      hi_val[18],
      hi_val[19],
      hi_val[20],
      hi_val[21],
      hi_val[22],
      hi_val[23],
      hi_val[24],
      hi_val[25],
      hi_val[26],
      hi_val[27],
      hi_val[28],
      hi_val[29],
      hi_val[30],
      hi_val[31]);
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
