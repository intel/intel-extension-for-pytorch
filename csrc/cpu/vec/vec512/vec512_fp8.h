#pragma once
#include <immintrin.h>
#include <cstdlib>
#include "utils/SysUtil.h"
#include "vec512_bfloat16.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {

static IPEX_FORCE_INLINE __m512i _mm512_cvte5m2_fp16(__m256i a) {
  return _mm512_slli_epi16(_mm512_cvtepi8_epi16(a), 8);
}

static IPEX_FORCE_INLINE __m512 _mm512_cvtpbh_ps(__m256i x) {
  return (__m512)_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 0x10);
}

static IPEX_FORCE_INLINE void cvt_fp32_e5m2_rne_intrinsic(
    const float* __restrict__ in,
    at::Float8_e5m2* out,
    int64_t len) {
  int64_t i = 0;
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  for (; i < len - 31; i += 32) {
    __m512 b = _mm512_loadu_ps(&in[i]);
    __m512 a = _mm512_loadu_ps(&in[i + 16]);

    __m256i ah_ =
        _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i bh_ =
        _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m512i a_ = _mm512_inserti64x4(
        _mm512_inserti64x4(_mm512_setzero_si512(), bh_, 0), ah_, 1);
    const __mmask32 maska1_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vnaninf), vnaninf, _MM_CMPINT_NE);
    const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
    __m512i a_rne_ = _mm512_mask_add_epi16(
        a_,
        maska1_,
        a_,
        _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
    a_rne_ = _mm512_srli_epi16(a_rne_, 8);
    _mm256_storeu_epi8(&out[i], _mm512_cvtepi16_epi8(a_rne_));
  }

  for (; i < len; i++) {
    out[i] = static_cast<at::Float8_e5m2>(in[i]);
  }
}

static IPEX_FORCE_INLINE void cvt_bf16_e5m2_rne_intrinsic(
    const at::BFloat16* __restrict__ in,
    at::Float8_e5m2* out,
    int64_t len) {
  int64_t i = 0;
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  for (; i < len - 31; i += 32) {
    __m512i x0 = _mm512_loadu_si512(&in[i]);
    __m512 b = _mm512_cvtpbh_ps(_mm512_extracti32x8_epi32(x0, 0));
    __m512 a = _mm512_cvtpbh_ps(_mm512_extracti32x8_epi32(x0, 1));

    __m256i ah_ =
        _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i bh_ =
        _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m512i a_ = _mm512_inserti64x4(
        _mm512_inserti64x4(_mm512_setzero_si512(), bh_, 0), ah_, 1);
    const __mmask32 maska1_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vnaninf), vnaninf, _MM_CMPINT_NE);
    const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
        _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
    __m512i a_rne_ = _mm512_mask_add_epi16(
        a_,
        maska1_,
        a_,
        _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
    a_rne_ = _mm512_srli_epi16(a_rne_, 8);
    _mm256_storeu_epi8(&out[i], _mm512_cvtepi16_epi8(a_rne_));
  }

  for (; i < len; i++) {
    out[i] = static_cast<at::Float8_e5m2>(in[i]);
  }
}

inline __m512i cvt_e4m3_bf16_intrinsic_without_denorm(__m256i fp8_vec) {
  // The following conversion is without denorm behavior, that is to say,
  //   Max subnorm   : S.0000.111 = 0.875 ∗ 2**(−6)
  //   Min subnorm   : S.0000.001 = 2**(−9)
  // 0.0019 ~ 0.0137 cannot be converted correctly.
  __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
  auto mask = _mm512_cmpneq_epi16_mask(
      _mm512_and_si512(x, _mm512_set1_epi16(127)),
      _mm512_setzero_si512()); // mask = x & 0x7f
  auto mask_nan = _mm512_cmpneq_epi16_mask(
      _mm512_and_si512(x, _mm512_set1_epi16(127)),
      _mm512_set1_epi16(127)); // mask_nan = x & 0x7f
  auto mantissa = _mm512_slli_epi16(
      _mm512_and_si512(x, _mm512_set1_epi16(7)), 4); // mantissa = (x & 7) << 4
  auto exponent = _mm512_add_epi16(
      _mm512_srli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(120)), 3),
      _mm512_set1_epi16(120)); // exponent = (((x >> 3) & 15) + 120)
  auto nonsign = _mm512_maskz_mov_epi16(
      mask, _mm512_or_si512(mantissa, _mm512_slli_epi16(exponent, 7)));
  nonsign = _mm512_mask_mov_epi16(
      _mm512_set1_epi16(0x7fff), mask_nan, nonsign); // deal with Nan
  return _mm512_or_si512(
      nonsign,
      _mm512_slli_epi16(
          _mm512_and_si512(x, _mm512_set1_epi16(128)),
          8)); // add sign (x & 128) << 8
}

inline __m512i cvt_e4m3_bf16_intrinsic_with_denorm(__m256i fp8_vec) {
  __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
  __m512i lg2mant = _mm512_mask_mov_epi16(
      _mm512_mask_mov_epi16(
          _mm512_setzero_si512(),
          _mm512_test_epi16_mask(x, _mm512_set1_epi16(2)),
          _mm512_set1_epi16(1)),
      _mm512_test_epi16_mask(x, _mm512_set1_epi16(4)),
      _mm512_set1_epi16(2));
  return _mm512_or_si512(
      _mm512_maskz_mov_epi16(
          _mm512_cmpneq_epi16_mask(
              _mm512_and_si512(x, _mm512_set1_epi16(127)),
              _mm512_setzero_si512()),
          _mm512_mask_blend_epi16(
              _mm512_test_epi16_mask(x, _mm512_set1_epi16(120)),
              _mm512_or_si512(
                  _mm512_and_si512(
                      _mm512_sllv_epi16(
                          _mm512_and_si512(x, _mm512_set1_epi16(3)),
                          _mm512_sub_epi16(_mm512_set1_epi16(7), lg2mant)),
                      _mm512_set1_epi16(0x007f)),
                  _mm512_slli_epi16(
                      _mm512_add_epi16(lg2mant, _mm512_set1_epi16(118)), 7)),
              _mm512_or_si512(
                  _mm512_slli_epi16(
                      _mm512_and_si512(x, _mm512_set1_epi16(7)), 4),
                  _mm512_slli_epi16(
                      _mm512_add_epi16(
                          _mm512_srli_epi16(
                              _mm512_and_si512(x, _mm512_set1_epi16(120)), 3),
                          _mm512_set1_epi16(120)),
                      7)))),
      _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(128)), 8));
}

static IPEX_FORCE_INLINE void cvt_e4m3_bf16_intrinsic(
    const at::Float8_e4m3fn* __restrict__ in,
    at::BFloat16* out,
    int64_t len,
    bool with_denorm = true) {
  int64_t i = 0;
  for (; i < len - 31; i += 32) {
    __m256i x0 = _mm256_loadu_si256((__m256i*)&in[i]);
    __m512i bh;
    if (with_denorm) {
      bh = cvt_e4m3_bf16_intrinsic_with_denorm(x0);
    } else {
      bh = cvt_e4m3_bf16_intrinsic_without_denorm(x0);
    }
    _mm512_storeu_si512((__m512i*)&out[i], bh);
  }
  for (; i < len; i++) {
    out[i] = static_cast<at::BFloat16>(in[i]);
  }
}

alignas(64) static uint16_t e4m3_to_16bit[256];

template <typename T>
static void initialize_e4m3_to_16bit_tables() {
  static bool initialized_16bit = false;
  if (!initialized_16bit) {
    for (uint8_t u8 = 0; u8 < 256; ++u8) {
      auto value = static_cast<T>(c10::bit_cast<c10::Float8_e4m3fn>(u8));
      uint16_t value_bits = c10::bit_cast<uint16_t>(value);
      e4m3_to_16bit[u8] = value_bits;
      if (u8 == 255)
        break;
    }
    initialized_16bit = true;
  }
}

template <typename T>
static IPEX_FORCE_INLINE void cvt_e4m3_16bit_intrinsic_lut(
    const at::Float8_e4m3fn* __restrict__ in,
    T* out,
    int64_t len) {
  for (size_t i = 0; i < len; i += 64) {
    __m512i fp8_vec = _mm512_loadu_si512((__m512i*)&in[i]);
    __m128i group0 = _mm512_castsi512_si128(fp8_vec);
    __m128i group1 = _mm512_extracti32x4_epi32(fp8_vec, 1);
    __m128i group2 = _mm512_extracti32x4_epi32(fp8_vec, 2);
    __m128i group3 = _mm512_extracti32x4_epi32(fp8_vec, 3);

    __m512i indices0 = _mm512_cvtepu8_epi32(group0);
    __m512i indices1 = _mm512_cvtepu8_epi32(group1);
    __m512i indices2 = _mm512_cvtepu8_epi32(group2);
    __m512i indices3 = _mm512_cvtepu8_epi32(group3);

    // Gather BF16 conversion results from the lookup table.
    __m512i bf16_i32_vec0 = _mm512_i32gather_epi32(indices0, e4m3_to_16bit, 2);
    __m512i bf16_i32_vec1 = _mm512_i32gather_epi32(indices1, e4m3_to_16bit, 2);
    __m512i bf16_i32_vec2 = _mm512_i32gather_epi32(indices2, e4m3_to_16bit, 2);
    __m512i bf16_i32_vec3 = _mm512_i32gather_epi32(indices3, e4m3_to_16bit, 2);

    if constexpr (std::is_same<T, float>()) {
      _mm512_storeu_si512(
          (__m512i*)(out + i + 0), _mm512_slli_epi32(bf16_i32_vec0, 16));
      _mm512_storeu_si512(
          (__m512i*)(out + i + 16), _mm512_slli_epi32(bf16_i32_vec1, 16));
      _mm512_storeu_si512(
          (__m512i*)(out + i + 32), _mm512_slli_epi32(bf16_i32_vec2, 16));
      _mm512_storeu_si512(
          (__m512i*)(out + i + 48), _mm512_slli_epi32(bf16_i32_vec3, 16));
    } else {
      // Helper lambda: Convert 16 32-bit ints (in a __m512i) to 16 16-bit ints.
      auto convert_32_to_16 = [](__m512i vec) -> __m256i {
        return _mm512_cvtepi32_epi16(vec);
      };

      __m256i bf16_i16_vec0 = convert_32_to_16(bf16_i32_vec0);
      __m256i bf16_i16_vec1 = convert_32_to_16(bf16_i32_vec1);
      __m256i bf16_i16_vec2 = convert_32_to_16(bf16_i32_vec2);
      __m256i bf16_i16_vec3 = convert_32_to_16(bf16_i32_vec3);

      _mm256_storeu_si256((__m256i*)(out + i + 0), bf16_i16_vec0);
      _mm256_storeu_si256((__m256i*)(out + i + 16), bf16_i16_vec1);
      _mm256_storeu_si256((__m256i*)(out + i + 32), bf16_i16_vec2);
      _mm256_storeu_si256((__m256i*)(out + i + 48), bf16_i16_vec3);
    }
  }
}

inline __m512i cvt_e4m3_fp32_intrinsic(__m128i fp8_vec) {
  // The following conversion is without denorm behavior, that is to say,
  //   Max subnorm   : S.0000.111 = 0.875 ∗ 2**(−6)
  //   Min subnorm   : S.0000.001 = 2**(−9)
  // 0.0019 ~ 0.0137 cannot be converted correctly.
  __m512i x = _mm512_cvtepu8_epi32(fp8_vec);
  auto mask = _mm512_cmpneq_epi32_mask(
      _mm512_and_si512(x, _mm512_set1_epi32(127)),
      _mm512_setzero_si512()); // mask = x & 0x7f
  auto mantissa = _mm512_slli_epi32(
      _mm512_and_si512(x, _mm512_set1_epi32(7)), 20); // mant = (x & 0x7) << 20
  auto exponent = _mm512_add_epi32(
      _mm512_and_si512(_mm512_srli_epi32(x, 3), _mm512_set1_epi32(15)),
      _mm512_set1_epi32(120)); // x >> 3 & 0xf + 120
  auto nonsign = _mm512_maskz_mov_epi32(
      mask,
      _mm512_or_si512(
          mantissa,
          _mm512_slli_epi32(exponent, 23))); // mant + (exponent << 23)
  return _mm512_or_si512(
      nonsign,
      _mm512_slli_epi32(_mm512_and_si512(x, _mm512_set1_epi32(128)), 24));
}

static IPEX_FORCE_INLINE void cvt_e4m3_fp32_intrinsic(
    const at::Float8_e4m3fn* __restrict__ in,
    float* out,
    int64_t len) {
  int64_t i = 0;
  for (; i < len - 15; i += 16) {
    __m128i x0 = _mm_loadu_si128((__m128i*)&in[i]);
    __m512i bh = cvt_e4m3_fp32_intrinsic(x0);
    _mm512_storeu_si512((__m512i*)&out[i], bh);
  }
  for (; i < len; i++) {
    out[i] = static_cast<float>(in[i]);
  }
}

static IPEX_FORCE_INLINE void cvt_e5m2_fp16_intrinsic(
    const at::Float8_e5m2* __restrict__ in,
    at::Half* out,
    int64_t len) {
  int64_t i = 0;
  for (; i < len - 31; i += 32) {
    __m256i x0 = _mm256_loadu_si256((__m256i*)&in[i]);
    __m512i ph = _mm512_cvte5m2_fp16(x0);
    _mm512_storeu_si512((__m512i*)&out[i], ph);
  }
  for (; i < len; i++) {
    out[i] = static_cast<at::Half>(in[i]);
  }
}

static IPEX_FORCE_INLINE void cvt_e5m2_bf16_intrinsic(
    const at::Float8_e5m2* __restrict__ in,
    at::BFloat16* out,
    size_t len) {
  int64_t i = 0;
  for (; i < len - 31; i += 32) {
    __m512i a = _mm512_cvte5m2_fp16(_mm256_loadu_epi8(&in[i]));
    __m256i ah = _mm512_extracti64x4_epi64(a, 0);
    __m256i bh = _mm512_extracti64x4_epi64(a, 1);
    __m256i a_ = cvt_fp32_to_bf16(_mm512_cvtph_ps(ah));
    __m256i b_ = cvt_fp32_to_bf16(_mm512_cvtph_ps(bh));
    _mm256_storeu_si256((__m256i*)(out + i), a_);
    _mm256_storeu_si256((__m256i*)(out + i + 16), b_);
  }
  for (; i < len; i++) {
    out[i] = static_cast<at::BFloat16>(in[i]);
  }
}

static IPEX_FORCE_INLINE void cvt_e5m2_fp32_intrinsic(
    const at::Float8_e5m2* __restrict__ in,
    float* out,
    size_t len) {
  int64_t i = 0;
  for (; i < len - 31; i += 32) {
    __m512i a = _mm512_cvte5m2_fp16(_mm256_loadu_epi8(&in[i]));
    __m256i ah = _mm512_extracti64x4_epi64(a, 0);
    __m256i bh = _mm512_extracti64x4_epi64(a, 1);
    __m512 a_ = _mm512_cvtph_ps(ah);
    __m512 b_ = _mm512_cvtph_ps(bh);
    _mm512_storeu_ps((out + i), a_);
    _mm512_storeu_ps((out + i + 16), b_);
  }
  for (; i < len; i++) {
    out[i] = static_cast<float>(in[i]);
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
