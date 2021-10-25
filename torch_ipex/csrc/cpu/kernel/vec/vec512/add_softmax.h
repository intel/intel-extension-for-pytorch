#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>

#include <limits>

namespace torch_ipex {
namespace cpu {
namespace kernel {
namespace vec {
namespace vec512 {

#define VEC_512_FP32_CAP (16)
#define VEC_512_FP32_BYTES_WIDTH (64)
#define FP32_BYTES_WIDTH (4)

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

static inline uint32_t _cal_head_padding(const float* addr) {
  uint64_t time_64 = ((uint64_t)addr) / VEC_512_FP32_BYTES_WIDTH;
  return (((uint64_t)addr) - time_64 * VEC_512_FP32_BYTES_WIDTH) /
      FP32_BYTES_WIDTH;
}

static inline uint32_t _cal_valid_data_num(
    const uint32_t& head_padding,
    const uint32_t& dim_size) {
  return std::min(VEC_512_FP32_CAP - head_padding, (uint32_t)dim_size);
}

static inline uint32_t _cal_tail_padding(
    const uint32_t& dim_size,
    const uint32_t& head_padding,
    const uint32_t& valid_data_num) {
  return VEC_512_FP32_CAP - head_padding - valid_data_num;
}

/**
 * Check if the start address is aligned or not. If the start address is not
 * 64 bytes aligned, we will pad head to align 64bytes and then pad tail to
 * fill 64bytes.
 */
static inline bool _padding_alignment(
    const float* a,
    const uint32_t& size,
    uint32_t& valid_data_num,
    __mmask16& loading_mask) {
  uint32_t head_padding = _cal_head_padding(a);
  if (head_padding == 0)
    return false;

  valid_data_num = _cal_valid_data_num(head_padding, (uint32_t)size);
  uint32_t tail_padding = _cal_tail_padding(size, head_padding, valid_data_num);
  loading_mask = ((1 << valid_data_num) - 1) << tail_padding;
  return true;
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

inline void _dil_add_reduce_max_fusion_kernel(
    const float* a,
    const float* b,
    const int& size,
    float* out,
    float& max) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_ps_min_tail = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_a = vec_ps_min;
  auto vec_b = vec_ps_min;
  auto vec_out = vec_ps_min;

  // Check if the start address is not aligned. If the start address is not
  // 64bytes aligned, we will pad head to align 64bytes and then pad tail to
  // fill 64bytes.
  uint32_t valid_data_num = 0;
  __mmask16 loading_mask = {};
  if (_padding_alignment(a, size, valid_data_num, loading_mask)) {
    vec_a = _mm512_maskz_expandloadu_ps(loading_mask, a);
    vec_b = _mm512_mask_expandloadu_ps(vec_ps_min, loading_mask, b);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_mask_compressstoreu_ps(out, loading_mask, vec_out);
  }

  int i = valid_data_num;
  for (; i <= size - 16; i += 16) {
    vec_a = _mm512_load_ps(a + i);
    vec_b = _mm512_load_ps(b + i);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_max_ps(vec_ps_min, vec_out);
    _mm512_store_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _mm512_mask_load_ps(vec_ps_min_tail, mask, a + i);
    vec_b = _mm512_maskz_load_ps(mask, b + i);
    vec_out = _mm512_add_ps(vec_a, vec_b);
    vec_ps_min = _mm512_max_ps(vec_out, vec_ps_min);
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

  // The start address is not aligned
  uint32_t valid_data_num = 0;
  __mmask16 loading_mask = {};
  if (_padding_alignment(a, size, valid_data_num, loading_mask)) {
    vec_a = _mm512_maskz_expandloadu_ps(loading_mask, a);
    vec_out = _mm512_sub_ps(vec_a, vec_max);
    vec_out = _dil_exp_kernel(vec_out);
    vec_sum = _mm512_mask_add_ps(vec_sum, loading_mask, vec_sum, vec_out);
    _mm512_mask_compressstoreu_ps(out, loading_mask, vec_out);
  }

  int i = valid_data_num;
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

inline void _dil_normalization_kernel(
    const float* a,
    const float& sum,
    const int& size,
    float* out) {
  auto vec_sum = _mm512_set1_ps(sum);
  __m512 vec_a = {};
  __m512 vec_out = {};

  // The start address is not aligned
  uint32_t valid_data_num = 0;
  __mmask16 loading_mask = {};
  if (_padding_alignment(a, size, valid_data_num, loading_mask)) {
    vec_a = _mm512_maskz_expandloadu_ps(loading_mask, a);
    vec_out = _mm512_div_ps(vec_a, vec_sum);
    _mm512_mask_compressstoreu_ps(out, loading_mask, vec_out);
  }

  int i = valid_data_num;
  for (; i <= size - 16; i += 16) {
    auto vec_a = _mm512_load_ps(a + i);
    auto vec_out = _mm512_div_ps(vec_a, vec_sum);
    _mm512_store_ps(out + i, vec_out);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_a = _mm512_maskz_load_ps(mask, a + i);
    auto vec_out = _mm512_div_ps(vec_a, vec_sum);
    _mm512_mask_store_ps(out + i, mask, vec_out);
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
 *
 * @param[in] a a contiguous tensor to be added
 * @param[in] b a tensor to be added while it should be broadcastable
 * @return The tensor stores the result of @code softmax(a + b) @endcode
 */
at::Tensor dil_add_softmax(const at::Tensor& a, const at::Tensor& b) {
  float* a_data_base = a.data_ptr<float>();
  float* b_data_base = b.data_ptr<float>();

  // Check if the tensor needs to be broadcasted
  auto infered_size = a.sizes().vec();
  auto need_broadcast = (infered_size != b.sizes());
  if (need_broadcast) {
    infered_size = at::infer_size(a.sizes(), b.sizes());
  }

  // Create an new tensor to store the output
  auto output = at::empty_like(a);
  float* output_data_base = output.data_ptr<float>();

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
      _dil_add_reduce_max_fusion_kernel(
          a_data_base + i * dim_size,
          b_data_base + b_offset,
          dim_size,
          output_data_base + i * dim_size,
          val);
      // Calculate the e^x and get the sum value:
      //    output_data = output_data - max(output_data)
      //    output_data = e^(output_data)
      //    val = sum(output_data)
      _dil_exp_reduce_sum_fusion_kernel(
          output_data_base + i * dim_size,
          dim_size,
          output_data_base + i * dim_size,
          val);
      // Calculat the normalization [e^x / sum(e^x)]:
      //    output_data = output_data / sum(output_data)
      _dil_normalization_kernel(
          output_data_base + i * dim_size,
          val,
          dim_size,
          output_data_base + i * dim_size);
    }
  });

  return output;
}

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
