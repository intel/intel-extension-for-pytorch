#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <limits>

#include "cpu/bf16/vec/vec_type_cvt.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {
namespace vec {
namespace vec512 {

__m512 _load_f32_data(const float* data_base) {
  return _mm512_load_ps(data_base);
}

__m512 _load_f32_data(const at::BFloat16* data_base) {
  return cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)data_base));
}

__m512 _maskz_load_f32_data(const float* data_base, __mmask16 mask) {
  return _mm512_maskz_load_ps(mask, data_base);
}

__m512 _maskz_load_f32_data(const at::BFloat16* data_base, __mmask16 mask) {
  return cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, (__m256i*)data_base));
}

void _store_data(float* data_base, __m512 a) {
  _mm512_store_ps(data_base, a);
}

void _store_data(at::BFloat16* data_base, __m512 a) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_store_si256((__m256i*)data_base, vec_bf16_out);
}

void _mask_store_data(float* data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_store_ps(data_base, mask, a);
}

void _mask_store_data(at::BFloat16* data_base, __m512 a, __mmask16 mask) {
  auto vec_bf16_out = cvt_fp32_to_bf16(a);
  _mm256_mask_storeu_epi16(data_base, mask, vec_bf16_out);
}

inline std::vector<int64_t> _adjust_strides(
    const at::Tensor& src,
    std::vector<int64_t>& infered_size) {
  // We does NOT support broadcasting last dim which mean last_dim = 1
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src.stride(src.ndimension() - 1) == 1);

  auto original_shape = src.sizes();
  auto original_stride = src.strides();
  auto offset = infered_size.size() - original_shape.size();

  std::vector<int64_t> adjusted_stride;
  if (offset > 0)
    adjusted_stride.resize(infered_size.size(), 0);
  else
    adjusted_stride.resize(infered_size.size());

  for (size_t i = 0; i < original_shape.size(); i++) {
    // see NOTE: [Computing output strides]
    if (original_shape[i] == 1 && infered_size[offset + i] != 1) {
      adjusted_stride[offset + i] = 0;
    } else {
      adjusted_stride[offset + i] = original_stride[i];
    }
  }

  return adjusted_stride;
}


inline int64_t _calc_element_offset(
    const int64_t& outer_loop_idx,
    const std::vector<int64_t>& outer_loop_size,
    const std::vector<int64_t>& outer_loop_strides) {
  int64_t __outer_loop_idx = outer_loop_idx;
  int64_t b_offset = 0;
  for (int j = 0; j < outer_loop_size.size(); j++) {
    auto idx = __outer_loop_idx / outer_loop_size[j];
    __outer_loop_idx -= idx * outer_loop_size[j];
    // The stride could be any number if the dim equals to 1
    b_offset += idx * outer_loop_strides[j];
  }
  return b_offset;
}

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
inline void _dil_add_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t* b,
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
    vec_a = _load_f32_data(a + i);
    vec_b = _load_f32_data(b + i);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_store_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_load_f32_data(a + i, mask);
    vec_b = _maskz_load_f32_data(b + i, mask);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_mask_max_ps(vec_ps_min, mask, vec_out, vec_ps_min);
    _mm512_mask_store_ps(out + i, mask, vec_out);
  }

  // NOTE: _mm512_reduce_max_ps is sequence instruction
  max = _mm512_reduce_max_ps(vec_ps_min);
}

inline void _dil_exp_reduce_sum_fusion_kernel(
    const float* a,
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
    vec_a = _mm512_load_ps(a + i);
    vec_out = _mm512_sub_ps(vec_a, vec_max);
    vec_out = _dil_exp_kernel(vec_out);
    vec_sum = _mm512_add_ps(vec_sum, vec_out);
    _mm512_store_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_a = _mm512_mask_load_ps(vec_max, mask, a + i);
    auto vec_out = _mm512_sub_ps(vec_a, vec_max);
    vec_out = _dil_exp_kernel(vec_out);
    vec_sum = _mm512_mask_add_ps(vec_sum, mask, vec_sum, vec_out);
    _mm512_mask_store_ps(out + i, mask, vec_out);
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
    auto vec_a = _mm512_load_ps(a + i);
    auto vec_out = _mm512_div_ps(vec_a, vec_sum);
    _store_data(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_a = _mm512_maskz_load_ps(mask, a + i);
    auto vec_out = _mm512_div_ps(vec_a, vec_sum);
    _mask_store_data(out + i, vec_out, mask);
  }
}

/**
 * @brief Fuse the add operator and softmax operator.
 *
 * @attention
 * There are some assumptions for this operator.
 * - The reduce dimension for softmax is the last dimension
 * - The reduce dimension for softmax is the leading dimension
 * - The elements number of the reduce dimension for softmax is n*16
 * - The input tensors are contiguous
 * - The number of the input tensor dimension should be >=2
 * - Only the second input tensor is brodcastable
 * - The datatype for inpusts(a,b) and output are same.
 *
 * @param[in] a a contiguous tensor to be added
 * @param[in] b a tensor to be added while it should be broadcastable
 * @return The tensor stores the result of @code softmax(a + b) @endcode
 */
template <typename scalar_t>
at::Tensor dil_add_softmax(const at::Tensor& a, const at::Tensor& b) {
  scalar_t* a_data_base = a.data_ptr<scalar_t>();
  scalar_t* b_data_base = b.data_ptr<scalar_t>();

  // Check if the tensor needs to be broadcasted
  auto infered_size = a.sizes().vec();
  auto need_broadcast = (infered_size != b.sizes());
  if (need_broadcast) {
    infered_size = at::infer_size(a.sizes(), b.sizes());
  }
  at::Tensor output = at::empty_like(a);
  // Create an new tensor to store the output
  scalar_t* output_data_base = output.data_ptr<scalar_t>();

  // Calculate the strides for the input tensor
  std::vector<int64_t> b_adjusted_strides = _adjust_strides(b, infered_size);

  std::vector<int64_t> outer_size_per_dim;
  int64_t dim_size = infered_size[infered_size.size() - 1];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim_size != 1);

  int64_t outer_size = 1;
  // The last dim is the loop unit. We need to minus 2 to exclude the last dim.
  // infered_size.size() - 2 is the -2th dimension.
  for (int64_t i = infered_size.size() - 2; i >= 0; i--) {
    // Record outer dimensions
    outer_size_per_dim.insert(outer_size_per_dim.begin(), outer_size);
    // Calculate outer loop number;
    outer_size *= infered_size[i];
  }

  int64_t grain_size = at::internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  int64_t outer_dims_num = outer_size_per_dim.size();
  at::parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
    float val = 0.0;
    int64_t b_offset = 0;
    at::Tensor tmp_out = at::empty({dim_size});
    float* tmp_out_ptr = tmp_out.data_ptr<float>();
    for (int64_t i = begin; i < end; i++) {
      if (need_broadcast) {
        b_offset =
            _calc_element_offset(i, outer_size_per_dim, b_adjusted_strides);
      } else {
        b_offset = i * dim_size;
      }
      // Add a and b and get the maximum value:
      //    output_data = a + b
      //    val = max(output_data)
      _dil_add_reduce_max_fusion_kernel<scalar_t>(
          a_data_base + i * dim_size,
          b_data_base + b_offset,
          dim_size,
          tmp_out_ptr,
          val);
      // Calculate the e^x and get the sum value:
      //    output_data = output_data - max(output_data)
      //    output_data = e^(output_data)
      //    val = sum(output_data)
      _dil_exp_reduce_sum_fusion_kernel(
          tmp_out_ptr, dim_size, tmp_out_ptr, val);
      // Calculat the normalization [e^x / sum(e^x)]:
      //    output_data = output_data / sum(output_data)
      _dil_normalization_kernel<scalar_t>(
          tmp_out_ptr, val, dim_size, output_data_base + i * dim_size);
    }
  });
  return output;
} // dil_add_softmax

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
