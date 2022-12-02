#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <limits>
#include "utils.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {

inline __m512 _dil_exp_kernel(__m512 vec_src) {
  static __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f); // 1/factorial(1)
  static __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f); // 1/factorial(2)
  static __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f); // 1/factorial(3)
  static __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
  static __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f); // 1/factorial(5)
  static __m512 vec_exp_log2ef =
      (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)
  static __m512 vec_half = _mm512_set1_ps(0.5f);
  static __m512 vec_one = _mm512_set1_ps(1.f);
  static __m512 vec_zero = _mm512_set1_ps(0.f);
  static __m512 vec_two = _mm512_set1_ps(2.f);
  static __m512 vec_ln2f = (__m512)_mm512_set1_epi32(0x3f317218); // ln(2)
  static __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
  static __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
  static __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  static int n_mantissa_bits = 23;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(vec_src, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  vec_src = _mm512_min_ps(vec_src, vec_ln_flt_max);
  vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundps_epi32(
      vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
  vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = (__m512)vec_two_pow_n_i;
  vec_two_pow_n =
      _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ps(vec_res, vec_two);
  return vec_res;
}

template <typename scalar_t>
inline void _dil_div_add_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t* b,
    const float& dim_per_head,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_ps_min_tail = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_a = vec_ps_min;
  auto vec_b = vec_ps_min;
  auto vec_out = vec_ps_min;

  int i = 0;
  auto vec_r_dim_per_head = _mm512_set1_ps(1.0 / dim_per_head);
  for (; i <= size - 16; i += 16) {
    vec_a = _loadu(a + i);
    vec_b = _loadu(b + i);
    vec_out = _mm512_fmadd_ps(vec_a, vec_r_dim_per_head, vec_b);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_b = _maskz_loadu(b + i, mask);
    vec_out = _mm512_fmadd_ps(vec_a, vec_r_dim_per_head, vec_b);
    vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

template <typename scalar_t>
inline void _dil_maskedfill_div_max_fusion_kernel(
    const scalar_t* a,
    const float* b,
    const float& fill_value,
    const float& dim_per_head,
    const int& size,
    float* out,
    float& max) {
  auto vec_fill = _mm512_set1_ps(fill_value);
  auto vec_ps_min = vec_fill;
  auto mask_c = _mm512_set1_ps(1.0);
  auto vec_dim_per_head = _mm512_set1_ps(dim_per_head);

  auto vec_a = vec_ps_min;
  auto vec_b = vec_ps_min;
  auto vec_out = vec_ps_min;

  int i = 0;
  for (; i <= size - 16; i += 16) {
    vec_a = _loadu(a + i);
    vec_b = _loadu(b + i);
    __mmask16 fill_mask = _mm512_cmp_ps_mask(vec_b, mask_c, 12);
    vec_out = _mm512_mask_div_ps(vec_fill, fill_mask, vec_a, vec_dim_per_head);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_b = _maskz_loadu(b + i, mask);
    __mmask16 fill_mask = _mm512_cmp_ps_mask(vec_b, mask_c, 12);
    vec_out = _mm512_mask_div_ps(vec_fill, fill_mask, vec_a, vec_dim_per_head);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

inline void _dil_exp_reduce_sum_fusion_kernel(
    float* a,
    const int& size,
    float* out,
    float& val) {
  static auto vec_zero = _mm512_set1_ps(0.f);
  auto vec_max = _mm512_set1_ps(val);
  auto vec_sum = _mm512_set1_ps(0.f);
  __m512 vec_a = {};
  __m512 vec_out = {};

  int i = 0;
  for (; i <= size - 16; i += 16) {
    vec_a = _mm512_loadu_ps(a + i);
    vec_out = _mm512_sub_ps(vec_a, vec_max);
    vec_out = _dil_exp_kernel(vec_out);
    vec_sum = _mm512_add_ps(vec_sum, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_a = _mm512_mask_loadu_ps(vec_max, mask, a + i);
    auto vec_out = _mm512_sub_ps(vec_a, vec_max);
    vec_out = _dil_exp_kernel(vec_out);
    vec_sum = _mm512_mask_add_ps(vec_sum, mask, vec_sum, vec_out);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_add_ps is sequence instruction
  val = _mm512_reduce_add_ps(vec_sum);
}

template <typename scalar_t>
inline void _dil_normalization_kernel(
    const float* a,
    const float& sum,
    const int& size,
    scalar_t* out) {
  auto vec_sum = _mm512_set1_ps(sum);
  __m512 vec_a = {};
  __m512 vec_out = {};

  int i = 0;
  for (; i <= size - 16; i += 16) {
    auto vec_a = _mm512_loadu_ps(a + i);
    auto vec_out = _mm512_div_ps(vec_a, vec_sum);
    _storeu(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_a = _mm512_maskz_loadu_ps(mask, a + i);
    auto vec_out = _mm512_div_ps(vec_a, vec_sum);
    _mask_storeu(out + i, vec_out, mask);
  }
}

inline void _dil_add_reduce_max_fusion_kernel(
    float* a,
    const float* b,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_ps_min_tail = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_a = vec_ps_min;
  auto vec_b = vec_ps_min;
  auto vec_out = vec_ps_min;

  int i = 0;
  for (; i <= size - 16; i += 16) {
    vec_a = _loadu(a + i);
    vec_b = _loadu(b + i);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_b = _maskz_loadu(b + i, mask);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
