#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <torch/types.h>
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

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline __m512h _dil_exp_half_kernel(__m512h vec_src) {
  static __m512h vec_factorial_1 =
      _mm512_set1_ph((_Float16)(0.999999701f)); // 1/factorial(1)
  static __m512h vec_factorial_2 =
      _mm512_set1_ph((_Float16)(0.499991506f)); // 1/factorial(2)
  static __m512h vec_factorial_3 =
      _mm512_set1_ph((_Float16)(0.166676521f)); // 1/factorial(3)
  static __m512h vec_factorial_4 =
      _mm512_set1_ph((_Float16)(0.0418978221f)); // 1/factorial(4)
  static __m512h vec_factorial_5 =
      _mm512_set1_ph((_Float16)(0.00828929059f)); // 1/factorial(5)
  static __m512h vec_exp_log2ef = (__m512h)_mm512_set1_epi16(0x3dc5); // log2(e)
  static __m512h vec_half = _mm512_set1_ph((_Float16)(0.5f));
  static __m512h vec_one = _mm512_set1_ph((_Float16)(1.f));
  static __m512h vec_zero = _mm512_set1_ph((_Float16)(0.f));
  static __m512h vec_two = _mm512_set1_ph((_Float16)(2.f));
  static __m512h vec_ln2f = (__m512h)_mm512_set1_epi16(0x398c); // ln(2)
  static __m512h vec_ln_flt_min = (__m512h)_mm512_set1_epi16(0xc8da);
  static __m512h vec_ln_flt_max = (__m512h)_mm512_set1_epi16(0x498b);
  static __m512i vec_15 = _mm512_set1_epi16(0x000f);
  static int n_mantissa_bits = 10;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ph_mask(vec_src, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  vec_src = _mm512_min_ph(vec_src, vec_ln_flt_max);
  vec_src = _mm512_max_ph(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ph(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundph_epi16(
      vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi16_ph(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ph(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ph(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ph(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ph(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ph(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ph(vec_exp_poly, vec_res, vec_one);

  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ph(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtph_epi16(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi16(vec_exp_number_i, vec_15);
  vec_two_pow_n_i = _mm512_slli_epi16(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = (__m512h)vec_two_pow_n_i;
  vec_two_pow_n =
      _mm512_mask_blend_ph(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ph(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ph(vec_res, vec_two);
  return vec_res;
}
#endif

/**
 * Previously vec_ps_min was set to std::numeric_limits<float>::min(),
 * the smallest positive number (FLT_MIN). This was wrong for ReduceMax
 * if all the input elements are negative, which will lead to exponent
 * overflow. The correct initial number to compare the max value should
 * be std::numeric_limits<float>::lowest() (-FLT_MAX), thus this kernel
 * can generate the correct max value for negative inputs.
 **/
template <typename scalar_a, typename scalar_b>
inline void _dil_div_add_reduce_max_fusion_kernel(
    const scalar_a* a,
    const scalar_b* b,
    const float& dim_per_head,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
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

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void _dil_div_add_reduce_max_fusion_kernel_half(
    const at::Half* a,
    const at::Half* b,
    const float& dim_per_head,
    const int& size,
    at::Half* out,
    at::Half& max) {
  auto vec_ps_min = _mm512_set1_ph((at::Half)(-65504.0));
  auto vec_a = vec_ps_min;
  auto vec_b = vec_ps_min;
  auto vec_out = vec_ps_min;

  int i = 0;
  auto vec_r_dim_per_head = _mm512_set1_ph((at::Half)(1.0 / dim_per_head));
  for (; i <= size - 32; i += 32) {
    vec_a = _loadu_half(a + i);
    vec_b = _loadu_half(b + i);
    vec_out = _mm512_fmadd_ph(vec_a, vec_r_dim_per_head, vec_b);
    vec_ps_min = _mm512_max_ph(vec_ps_min, vec_out);
    _storeu_Half(out + i, vec_out);
  }

  if (i < size) {
    __mmask32 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_b = _maskz_loadu(b + i, mask);
    vec_out = _mm512_fmadd_ph(vec_a, vec_r_dim_per_head, vec_b);
    vec_ps_min = _mm512_mask_max_ph(vec_ps_min, mask, vec_out, vec_ps_min);
    _mask_storeu(out + i, vec_out, mask);
  }

  max = _mm512_reduce_max_ph(vec_ps_min);
}
#endif

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

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void _dil_exp_reduce_sum_fusion_kernel_half(
    at::Half* a,
    const int& size,
    at::Half* out,
    at::Half& val) {
  static auto vec_zero = _mm512_set1_ph((at::Half)(0.0));
  auto vec_max = _mm512_set1_ph(val);
  auto vec_sum = _mm512_set1_ph(at::Half(0.0));
  __m512h vec_a = {};
  __m512h vec_out = {};

  int i = 0;
  for (; i <= size - 32; i += 32) {
    vec_a = _loadu_half(a + i);
    vec_out = _mm512_sub_ph(vec_a, vec_max);
    vec_out = _dil_exp_half_kernel(vec_out);
    vec_sum = _mm512_add_ph(vec_sum, vec_out);
    _storeu_Half(out + i, vec_out);
  }

  if (i < size) {
    __mmask32 mask = (1 << (size - i)) - 1;
    auto vec_a = (__m512h)(_mm512_mask_loadu_epi16(
        (__m512i)(vec_max), mask, (__m512i*)(a + i)));
    auto vec_out = _mm512_sub_ph(vec_a, vec_max);
    vec_out = _dil_exp_half_kernel(vec_out);
    vec_sum = _mm512_mask_add_ph(vec_sum, mask, vec_sum, vec_out);
    _mask_storeu(out + i, vec_out, mask);
  }

  val = _mm512_reduce_add_ph(vec_sum);
}
#endif

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

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void _dil_normalization_kernel_half(
    const at::Half* a,
    const at::Half& sum,
    const int& size,
    at::Half* out) {
  auto vec_sum = _mm512_set1_ph(sum);
  __m512h vec_a = {};
  __m512h vec_out = {};

  int i = 0;
  for (; i <= size - 32; i += 32) {
    auto vec_a = _loadu_half(a + i);
    auto vec_out = _mm512_div_ph(vec_a, vec_sum);
    _storeu_Half(out + i, vec_out);
  }

  if (i < size) {
    __mmask32 mask = (1 << (size - i)) - 1;
    auto vec_a = _maskz_loadu(a + i, mask);
    auto vec_out = _mm512_div_ph(vec_a, vec_sum);
    _mask_storeu(out + i, vec_out, mask);
  }
}
#endif

template <typename scalar_t>
inline void _dil_add_kernel(const scalar_t* src, float* dst, const int& size) {
  __m512 vec_a = {};
  __m512 vec_out = {};

  int j = 0;
  for (; j <= size - 16; j += 16) {
    vec_a = _loadu(src + j);
    vec_out = _loadu(dst + j);
    vec_out = _mm512_add_ps(vec_a, vec_out);
    _storeu(dst + j, vec_out);
  }

  if (j < size) {
    __mmask16 mask = (1 << (size - j)) - 1;
    vec_a = _maskz_loadu(src + j, mask);
    vec_out = _maskz_loadu(dst + j, mask);
    vec_out = _mm512_add_ps(vec_out, vec_a);
    _mask_storeu(dst + j, vec_out, mask);
  }
}

inline void _dil_add_reduce_max_fusion_kernel(
    float* a,
    const float* b,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
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

inline void _dil_reduce_max_fusion_kernel(
    const float* a,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  auto vec_out = vec_ps_min;

  int i = 0;
  for (; i <= size - 16; i += 16) {
    vec_out = _loadu(a + i);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_out = _maskz_loadu(a + i, mask);
    vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

inline void _dil_mul_reduce_max_fusion_kernel(
    const float* a,
    const float& scale,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  auto vec_a = vec_ps_min;
  auto vec_out = vec_ps_min;

  int i = 0;
  auto vec_scale = _mm512_set1_ps(scale);
  for (; i <= size - 16; i += 16) {
    vec_a = _loadu(a + i);
    vec_out = _mm512_mul_ps(vec_a, vec_scale);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_out = _mm512_mul_ps(vec_a, vec_scale);
    vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

inline void _init_mha_buffer_kernel(float* max, float* sum, const int& size) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  auto vec_zeros = _mm512_setzero_ps();

  int i = 0;
  for (; i <= size - 16; i += 16) {
    _storeu(max + i, vec_ps_min);
    _storeu(sum + i, vec_zeros);
  }
  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    _mask_storeu(max + i, vec_ps_min, mask);
    _mask_storeu(sum + i, vec_zeros, mask);
  }
}

/**
 * This kernel is used to reorder the data type of the MHA output
 * from FP32 to BF16 with strides.
 * src: MKL BF16 GEMM output buffer, dtype - FP32
 * dst: Final MHA output, dtype - BF16
 */
template <typename scalar_t>
inline void _reorder_mha_output_kernel(
    float* src,
    scalar_t* dst,
    const int& rows,
    const int& cols,
    const int& dst_stride) {
  for (int i = 0; i < rows; ++i) {
    int j = 0;
    for (; j <= cols - 16; j += 16) {
      _storeu(dst + i * dst_stride + j, _loadu(src + i * cols + j));
    }
    if (j < cols) {
      __mmask16 mask = (1 << (cols - j)) - 1;
      _mask_storeu(dst + i * dst_stride + j, _loadu(src + i * cols + j), mask);
    }
  }
}

/**
 * This kernel is used to update the MHA output with the latest MAX
 * and SUM values block by block.
 * exp_val: exp(max_old - max_new)
 * In the i th block, the softmax(qk - i th) * v - i th was calculated
 * with the old MAX and SUM values, max_old and sum_old. When moving to
 * the i + 1 th block, since softmax(qk - i + 1 th) will be calculated
 * with the new MAX and SUM values, max_new and sum_new, thus the MHA
 * buffer which stores the summation of blocked softmax(qk) * v should
 * be also updated using max_new and sum_new:
 * a = a * sum_old / sum_new
 * a = a * exp(max_old) / exp(max_new) = a * exp_val
 */
inline void _mha_update_sum_max_kernel(
    const float* a,
    const float& sum_old,
    const float& sum_new,
    const float& exp_val,
    const int& size,
    float* out) {
  auto vec_sum_old = _mm512_set1_ps(sum_old);
  auto vec_sum_new = _mm512_set1_ps(sum_new);
  auto vec_sum_cor = _mm512_div_ps(vec_sum_old, vec_sum_new);
  auto exp_vec = _mm512_set1_ps(exp_val);

  int i = 0;
  for (; i <= size - 16; i += 16) {
    auto dat = _loadu(a + i);
    auto vec_a = _mm512_mul_ps(dat, vec_sum_cor);
    auto vec_out = _mm512_mul_ps(vec_a, exp_vec);
    _storeu(out + i, vec_out);
  }
  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto dat = _mm512_maskz_loadu_ps(mask, a + i);
    auto vec_a = _mm512_mul_ps(dat, vec_sum_cor);
    auto vec_out = _mm512_mul_ps(vec_a, exp_vec);
    _mask_storeu(out + i, vec_out, mask);
  }
}

template <typename scalar_a>
inline void _dil_div_add_alibi_and_reduce_max_fusion_kernel(
    const scalar_a* a,
    const float& scale,
    const int& size,
    float* out,
    float& max,
    const float alibi_slope,
    bool use_alibi) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  auto vec_a = vec_ps_min;
  auto vec_out = vec_ps_min;

  int i = 0;
  auto vec_scale = _mm512_set1_ps(scale);
  auto vec_alibi_slope = _mm512_set1_ps(alibi_slope);
  float idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto vec_idx = _mm512_loadu_ps(idx);
  for (; i <= size - 16; i += 16) {
    vec_a = _loadu(a + i);
    vec_out = _mm512_mul_ps(vec_a, vec_scale);
    if (use_alibi) {
      // alibi_slope * (token_idx - context_len +1)
      auto vec_token_idx = _mm512_set1_ps(i);
      vec_token_idx = _mm512_add_ps(vec_token_idx, vec_idx);
      auto vec_context_len = _mm512_set1_ps(size + 1);
      vec_token_idx = _mm512_sub_ps(vec_token_idx, vec_context_len);
      vec_token_idx = _mm512_mul_ps(vec_token_idx, vec_alibi_slope);
      vec_out = _mm512_add_ps(vec_out, vec_token_idx);
    }
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_storeu_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_out = _mm512_mul_ps(vec_a, vec_scale);
    if (use_alibi) {
      // alibi_slope * (token_idx - context_len +1)
      auto vec_token_idx = _mm512_set1_ps(i);
      vec_token_idx = _mm512_add_ps(vec_token_idx, vec_idx);
      auto vec_context_len = _mm512_set1_ps(size + 1);
      vec_token_idx = _mm512_sub_ps(vec_token_idx, vec_context_len);
      vec_token_idx = _mm512_mul_ps(vec_token_idx, vec_alibi_slope);
      vec_out = _mm512_add_ps(vec_out, vec_token_idx);
    }
    vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
    _mm512_mask_storeu_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

template <typename QT, typename KT, typename CT>
void _reduce_head(
    const QT* q_ptr_start,
    const KT* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    CT* k_cache_start) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto q_vec = _loadu(q_ptr_start + hsi);
    auto k_vec = _loadu(k_ptr_start + hsi);
    if (store_key) {
      _storeu(k_cache_start + hsi, k_vec);
    }
    qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] =
          (float)k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * (float)k_ptr_start[hsi];
  }
}

template <typename VT, typename OT, typename CT>
inline void _mul_and_accumulate(
    const float& attn_w,
    const VT* v_ptr_start,
    OT* attn_out_start,
    int64_t head_size,
    bool store_value,
    CT* v_cache_start,
    int accumulated) {
  auto vec_size = 16; // 512/32
  auto hsi = 0;
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto attn_w_vec = _mm512_set1_ps(attn_w);
    auto v_vec = _loadu(v_ptr_start + hsi);
    if (accumulated) {
      auto attn_out_vec = _loadu(attn_out_start + hsi);
      auto attn_out_vec_new = _mm512_fmadd_ps(attn_w_vec, v_vec, attn_out_vec);
      _storeu(attn_out_start + hsi, attn_out_vec_new);
    } else {
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec, v_vec);
      _storeu(attn_out_start + hsi, attn_out_vec_new);
    }
    if (store_value) {
      _storeu(v_cache_start + hsi, v_vec);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulated) {
      attn_out_start[hsi] += attn_w * (float)v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * (float)v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = (float)v_ptr_start[hsi];
    }
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
