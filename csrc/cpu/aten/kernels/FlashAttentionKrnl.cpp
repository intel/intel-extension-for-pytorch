#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>

#include <ATen/Tensor.h>
#include <aten/FlashAttention.h>
#include <aten/utils/mkl_gemm.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "../cpu/utils/isa_utils.h"
#include "csrc/cpu/tpp/woq/tla.h"
#include "vec/vec.h"

namespace torch_ipex {
using namespace tpp;
namespace cpu {

namespace {

// TODO: Use at::native::_store instead when it supports Half.
template <typename scalar_t>
inline void _store(scalar_t* dst, at::vec::Vectorized<scalar_t> src) {
  src.store(dst);
}

template <typename scalar_t>
inline typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, void>
_store(scalar_t* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_from_float<scalar_t>(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

template <typename scalar_t>
inline void pad_row_zero(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int i = 0;
  for (; i < rows - 1; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }

  // zero padding
  int j = 0;
  for (; j < cols - (cols % vec_size); j += vec_size) {
    auto vec_v = at::vec::Vectorized<scalar_t>(0);
    vec_v.store(padding_value_ptr + i * cols + j);
  }

  if (j < cols) {
    auto vec_v = at::vec::Vectorized<scalar_t>(0);
    vec_v.store(padding_value_ptr + i * cols + j, cols - j);
  }
}

template <typename scalar_t>
inline void pad_col_zero(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  for (int i = 0; i < rows; i++) {
    int j = 0;
    for (; j < cols - 1 - ((cols - 1) % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }
    if (j < cols - 1) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - 1 - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - 1 - j);
      *(padding_value_ptr + i * cols + cols - 1) = scalar_t(0);
    }
  }
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_INTERNAL_ASSERT(ptr2 == nullptr);
  return ptr;
}

template <
    typename scalar_t,
    typename std::enable_if_t<
        at::vec::CPU_CAPABILITY::is_reduced_floating_point_v<scalar_t>,
        int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

template <typename scalar_t>
inline Vectorized<scalar_t> exp_u20(Vectorized<scalar_t> data) {
  return data.exp_u20();
}
#if defined(CPU_CAPABILITY_AVX512)
// To implement exp_u20 here is faster than calling from add_softmax.h or PT
// vec512_float.h
inline Vectorized<float> exp_u20(Vectorized<float> data) {
  __m512 values = __m512(data);
  // A faster version of exp with ULP=20
  const __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f); // 1/factorial(1)
  const __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f); // 1/factorial(2)
  const __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f); // 1/factorial(3)
  const __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
  const __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f); // 1/factorial(5)
  const __m512 vec_exp_log2ef =
      (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)
  const __m512 vec_half = _mm512_set1_ps(0.5f);
  const __m512 vec_one = _mm512_set1_ps(1.f);
  const __m512 vec_zero = _mm512_set1_ps(0.f);
  const __m512 vec_two = _mm512_set1_ps(2.f);
  const __m512 vec_ln2f = (__m512)_mm512_set1_epi32(0x3f317218); // ln(2)
  const __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
  const __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
  const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  const int n_mantissa_bits = 23;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
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
#endif

// out = val * a + b
// is_b_stride_zero: If the stride of b is 0 (mask broadcasting case),
//                take b as a scalar pointer.
template <bool is_b_stride_zero, typename T1, typename T2>
inline void _scale_attn_mask_fusion_kernel(
    T1* a,
    T2* b,
    const int& size,
    T1* out,
    T1& val) {
  const auto vec_size1 = at::vec::Vectorized<T1>::size();
  const auto vec_size2 = at::vec::Vectorized<T2>::size();
  constexpr int64_t T1_n =
      (vec_size2 == vec_size1 * 2 && is_reduced_floating_point_v<T2>) ? 2 : 1;
  constexpr int64_t T2_n = 1;
  auto vec_scale = at::vec::VectorizedN<T1, T1_n>(val);
  int64_t i = 0;
  if (is_b_stride_zero) {
    auto b_first_val = (T1)b[0];
    auto b_first_vec = at::vec::VectorizedN<T2, T2_n>(b_first_val);
    for (; i < size - (size % vec_size2); i += vec_size2) {
      auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
      auto b_n = b_first_vec;
      at::vec::VectorizedN<T1, T1_n> b_n_convert =
          at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
      auto res = a_n * vec_scale + b_n_convert;
      res.store(out + i);
    }
    for (; i < size; i++) {
      auto tmp0 = a[i];
      auto tmp1 = b_first_val;
      out[i] = tmp0 * val + tmp1;
    }
  } else {
    for (; i < size - (size % vec_size2); i += vec_size2) {
      auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
      auto b_n = at::vec::VectorizedN<T2, T2_n>::loadu(b + i);
      at::vec::VectorizedN<T1, T1_n> b_n_convert =
          at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
      auto res = a_n * vec_scale + b_n_convert;
      res.store(out + i);
    }
    for (; i < size; i++) {
      auto tmp0 = a[i];
      auto tmp1 = (T1)b[i];
      out[i] = tmp0 * val + tmp1;
    }
  }
}

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = exp_u20(tmp1);
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      at::vec::vec_reduce_all<scalar_t>(
          [](at::vec::Vectorized<scalar_t>& x,
             at::vec::Vectorized<scalar_t>& y) {
            return at::vec::maximum(x, y);
          },
          vec_tmp_max));
}

// This function is used to produce an attn_mask in a standard format
inline std::optional<at::Tensor> convert_boolean_attn_mask(
    const std::optional<at::Tensor>& attn_mask,
    caffe2::TypeMeta dtype) {
  // Pass through
  if (!attn_mask.has_value()) {
    return c10::nullopt;
  }
  // Convert boolean mask to additive mask
  if (attn_mask->dtype() == at::kBool) {
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

// Support mask shapes:
// 2d: ({Q_seq_len, 1}  x {KV_seq_len, 1})
// 4d: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
inline bool check_attn_mask_shape(
    at::Tensor& attn_mask,
    int64_t batchSize,
    int64_t num_head,
    int64_t qSize,
    int64_t kvSize) {
  if (attn_mask.size(-2) != qSize && attn_mask.size(-2) != 1) {
    return false;
  }
  if (attn_mask.size(-1) != kvSize && attn_mask.size(-1) != 1) {
    return false;
  }
  if (attn_mask.dim() == 2) {
    return true;
  } else if (attn_mask.dim() == 4) {
    if ((attn_mask.size(0) == 1 || attn_mask.size(0) == batchSize) &&
        (attn_mask.size(1) == 1 || attn_mask.size(1) == num_head)) {
      return true;
    }
  }
  return false;
}

// Reshape attention mask to 4d
inline void reshape_attn_mask_to_4d(
    at::Tensor& attn_mask,
    int64_t batchSize,
    int64_t num_head,
    int64_t qSize,
    int64_t kvSize) {
  TORCH_CHECK(
      check_attn_mask_shape(attn_mask, batchSize, num_head, qSize, kvSize),
      "IPEX flash_attention: Please use the following attn mask shapes: ",
      "2d - ({Q_seq_len, 1}  x {KV_seq_len, 1}); ",
      "4d - ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})");
  int64_t attn_mask_size_0 = 1;
  int64_t attn_mask_size_1 = 1;
  if (attn_mask.dim() == 4) {
    if (attn_mask.size(0) == batchSize) {
      attn_mask_size_0 = batchSize;
    }
    if (attn_mask.size(1) == num_head) {
      attn_mask_size_1 = num_head;
    }
  }
  attn_mask = attn_mask
                  .view(
                      {attn_mask_size_0,
                       attn_mask_size_1,
                       attn_mask.size(-2),
                       attn_mask.size(-1)})
                  .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

/*
 *Caculate the flash attention SDPA.
 *@template scalar_t: q/k/v data type
 *@template q_split_size: q block size
 *@template kv_split_size: kv block size
 *@param output: output result
 *@param logsumexp: logsumexp for backward
 *@param q: query
 *@param k: key
 *@param v: value
 *@param dropout_p: dropout probability
 *@param is_causal: assume causal attention masking if true
 *@param attention_mask: attention mask
 *@param scale: scaling factor applied prior to softmax
 */
template <
    typename scalar_t,
    typename mask_t,
    int64_t q_split_size,
    int64_t kv_split_size>
inline typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, void>
cpu_flash_attention(
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  // reshape mask
  if (attention_mask.has_value()) {
    reshape_attn_mask_to_4d(
        attention_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t lStrideB = logsumexp.stride(0);
  int64_t lStrideM = logsumexp.stride(1);
  int64_t lStrideH = logsumexp.stride(2);
  int64_t mStrideB =
      (attention_mask.has_value() && attention_mask.value().size(0) > 1)
      ? attention_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (attention_mask.has_value() && attention_mask.value().size(1) > 1)
      ? attention_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (attention_mask.has_value() && attention_mask.value().size(2) > 1)
      ? attention_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (attention_mask.has_value() && attention_mask.value().size(3) > 1)
      ? attention_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::empty(
      {num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attention_mask.has_value()
      ? attention_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          torch_ipex::cpu::kernel::fill_stub(
              qk_max_data,
              -std::numeric_limits<accum_t>::infinity(),
              qBlockSize);
          torch_ipex::cpu::kernel::fill_stub(
              qk_sum_data, static_cast<accum_t>(0), qBlockSize);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            // Calculate scale * q @ k.T
            _mkl_gemm(
                CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                kvBlockSize,
                qBlockSize,
                headSize,
                static_cast<accum_t>(1),
                k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                kStrideN,
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                qStrideM,
                static_cast<accum_t>(0),
                qk_data,
                kvBlockSize);
            // Apply causal mask, fill unused with -inf
            if (is_causal && num_keys - n <= kvSplitSize) {
              for (const auto row : c10::irange(qBlockSize)) {
                int64_t last_col = m + row - n;
                accum_t* row_ptr = qk_data + row * kvBlockSize;
                torch_ipex::cpu::kernel::fill_stub(
                    row_ptr + last_col + 1,
                    -std::numeric_limits<accum_t>::infinity(),
                    kvBlockSize - last_col - 1);
              }
            }
            // Update attention weights with attention mask
            // And apply scaling factor
            if (attention_mask.has_value()) {
              for (int64_t row = 0; row < qBlockSize; ++row) {
                // qk <- qk * scaling_factor + attn_mask, else
                if (mStrideN == 0) {
                  _scale_attn_mask_fusion_kernel</*is_stride_zero*/ true>(
                      qk_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM,
                      kvBlockSize,
                      qk_data + row * kvBlockSize,
                      scaling_factor);
                } else {
                  _scale_attn_mask_fusion_kernel</*is_stride_zero*/ false>(
                      qk_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM + n,
                      kvBlockSize,
                      qk_data + row * kvBlockSize,
                      scaling_factor);
                }
              }
            }
            // Update coefficients with Softmax
            accum_t tmp_max = 0, tmp_sum = 0, sum_old = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
              sum_old = qk_sum_data[row];
              if (attention_mask.has_value()) {
                // max per row
                tmp_max = at::vec::reduce_all<accum_t>(
                    [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
                    qk_data + row * kvBlockSize,
                    kvBlockSize);
              } else {
                // apply scaling factor and max per row in fusion
                _mul_reduce_max_fusion_kernel(
                    qk_data + row * kvBlockSize,
                    scaling_factor,
                    kvBlockSize,
                    qk_data + row * kvBlockSize,
                    tmp_max);
              }
              tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
              // qk <- exp(qk - max) and sum per row
              tmp_sum = tmp_max;
              _exp_reduce_sum_fusion_kernel(
                  qk_data + row * kvBlockSize,
                  kvBlockSize,
                  qk_data + row * kvBlockSize,
                  tmp_sum);
              // exp_tmp <- exp(max[row] - max)
              exp_tmp = std::exp(qk_max_data[row] - tmp_max);
              // sum[row] <- sum + exp_tmp * sum[row]
              qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
              // max[row] <- max
              qk_max_data[row] = tmp_max;
              // dst <- dst * exp_tmp
              if (n > 0) {
                at::vec::map<accum_t>(
                    [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                    dst_data + row * headSize,
                    dst_data + row * headSize,
                    headSize);
              }
            }
            // Calculate Softmax(q @ k.T) @ v
            _mkl_gemm(
                CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                headSize,
                qBlockSize,
                kvBlockSize,
                static_cast<accum_t>(1),
                v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                vStrideN,
                qk_data,
                kvBlockSize,
                n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
                dst_data,
                headSize);
          }
          // dst <- dst / sum[row]
          // reorder MHA output with strides
          for (int64_t row = 0; row < qBlockSize; ++row) {
            accum_t sum_reciprocal = 1 / qk_sum_data[row];
            at::vec::map<scalar_t>(
                [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
                out_data + i * oStrideB + j * oStrideH + m * oStrideM +
                    row * oStrideM,
                dst_data + row * headSize,
                headSize);
          }
          // Store logsumexp for backward
          accum_t* lse_ptr =
              lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
          for (const auto row : c10::irange(qBlockSize)) {
            lse_ptr[row * lStrideM] =
                qk_max_data[row] + std::log(qk_sum_data[row]);
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
}

// Half/BFloat16
template <
    typename scalar_t,
    typename mask_t,
    int64_t q_split_size,
    int64_t kv_split_size>
inline typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, void>
cpu_flash_attention(
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);
  const bool is_fp16 = dtype == at::kHalf;
  const int vnni_pack =
      (!is_fp16 || (is_fp16 && utils::isa_has_amx_fp16_support())) ? 1 : 0;

  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  // reshape mask
  if (attention_mask.has_value()) {
    reshape_attn_mask_to_4d(
        attention_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t lStrideB = logsumexp.stride(0);
  int64_t lStrideM = logsumexp.stride(1);
  int64_t lStrideH = logsumexp.stride(2);
  int64_t mStrideB =
      (attention_mask.has_value() && attention_mask.value().size(0) > 1)
      ? attention_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (attention_mask.has_value() && attention_mask.value().size(1) > 1)
      ? attention_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (attention_mask.has_value() && attention_mask.value().size(2) > 1)
      ? attention_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (attention_mask.has_value() && attention_mask.value().size(3) > 1)
      ? attention_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::empty(
      {num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::empty(
      {num_thread,
       qSplitSize,
       kvSplitSize % 2 == 0 ? kvSplitSize : kvSplitSize + 1},
      query.options());
  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attention_mask.has_value()
      ? attention_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = buf_reduced.data_ptr<scalar_t>();

  // Create tpp kernels for Query @ Key
  bool headSize_even = headSize % 2 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int qk_gemm_K = headSize_even ? headSize : headSize + 1;

  auto qk_gemm = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qSplitSize,
      /*N*/ kvSplitSize,
      /*K*/ qk_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ headSize_even ? qStrideM : qk_gemm_K,
      /*ldb*/ kvSplitSize,
      /*ldc*/ kvSplitSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));
  auto qk_gemm_ktail = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qSplitSize,
      /*N*/ kvTail,
      /*K*/ qk_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ headSize_even ? qStrideM : qk_gemm_K,
      /*ldb*/ kvTail,
      /*ldc*/ kvTail,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));
  auto qk_gemm_qtail = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qTail,
      /*N*/ kvSplitSize,
      /*K*/ qk_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ headSize_even ? qStrideM : qk_gemm_K,
      /*ldb*/ kvSplitSize,
      /*ldc*/ kvSplitSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));
  auto qk_gemm_qktail = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qTail,
      /*N*/ kvTail,
      /*K*/ qk_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ headSize_even ? qStrideM : qk_gemm_K,
      /*ldb*/ kvTail,
      /*ldc*/ kvTail,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));

  // Create tpp kernels for Attention @ Value
  bool av_gemm_K_even = kvSplitSize % 2 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int av_gemm_K = av_gemm_K_even ? kvSplitSize : kvSplitSize + 1;
  bool av_gemm_K_tail_even = kvTail % 2 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int av_gemm_K_tail = av_gemm_K_tail_even ? kvTail : kvTail + 1;

  // [qSplitSize,kvSplitSize] x [kvSplitSize,headSize] -> [qSplitSize,headSize]
  auto av_gemm = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ vnni_pack ? av_gemm_K : kvSplitSize,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K,
      /*ldb*/ vnni_pack ? headSize : vStrideN,
      /*ldc*/ headSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));
  auto av_gemm_tail = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ vnni_pack ? av_gemm_K_tail : kvTail,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K_tail,
      /*ldb*/ vnni_pack ? headSize : vStrideN,
      /*ldc*/ headSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));
  auto av_gemm_bias = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ vnni_pack ? av_gemm_K : kvSplitSize,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K,
      /*ldb*/ vnni_pack ? headSize : vStrideN,
      /*ldc*/ headSize,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));
  auto av_gemm_bias_tail = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ vnni_pack ? av_gemm_K_tail : kvTail,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K_tail,
      /*ldb*/ vnni_pack ? headSize : vStrideN,
      /*ldc*/ headSize,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ vnni_pack)));

  // Buffer to store Key and Value after transforms
  at::Tensor key_t_reorder = at::empty(
      {batchSize,
       num_head,
       (!headSize_even && is_fp16) ? qk_gemm_K : headSize,
       kvSize},
      c10::CppTypeToScalarType<scalar_t>::value);
  // Buffer to store padding key and query
  scalar_t* key_padding_ptr = nullptr;
  std::unique_ptr<unsigned short[]> key_padding_data;
  scalar_t* query_padding_ptr = nullptr;
  std::unique_ptr<unsigned short[]> query_padding_data;
  if (!headSize_even && is_fp16) {
    query_padding_data = std::make_unique<unsigned short[]>(
        num_thread * (qSlice * qSplitSize) * qk_gemm_K);
    query_padding_ptr = reinterpret_cast<scalar_t*>(query_padding_data.get());
    key_padding_data = std::make_unique<unsigned short[]>(
        batchSize * num_head * kvSize * qk_gemm_K);
    key_padding_ptr = reinterpret_cast<scalar_t*>(key_padding_data.get());
  }

  auto key_reorder_ptr = key_t_reorder.data_ptr<scalar_t>();
  int kv_padding_size = (kvSize - 1) / kvSplitSize * av_gemm_K + av_gemm_K_tail;
  // Buffer to store padding value
  scalar_t* value_padding_ptr = nullptr;
  std::unique_ptr<unsigned short[]> value_padding_data;
  if (!av_gemm_K_even || !av_gemm_K_tail_even) {
    value_padding_data = std::make_unique<unsigned short[]>(
        batchSize * num_head * kv_padding_size * headSize);
    value_padding_ptr = reinterpret_cast<scalar_t*>(value_padding_data.get());
  }
  at::Tensor value_t_reorder = at::empty(
      {batchSize, num_head, kv_padding_size, headSize},
      c10::CppTypeToScalarType<scalar_t>::value);
  auto value_reorder_ptr = value_t_reorder.data_ptr<scalar_t>();

  // Create tpp transforms for Key
  auto k_xform = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ kvSplitSize,
          /*in_cols*/ qk_gemm_K,
          /*out_rows*/ qk_gemm_K,
          /*out_cols*/ kvSplitSize,
          /*ldi*/ headSize_even ? kStrideN : qk_gemm_K,
          /*ldo*/ kvSplitSize,
          /*xtype*/
          vnni_pack ? XformTPP::XFORM_XPOSE_N2V_TPP : XformTPP::XFORM_XPOSE_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);
  auto k_xform_tail = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ kvTail,
          /*in_cols*/ qk_gemm_K,
          /*out_rows*/ qk_gemm_K,
          /*out_cols*/ kvTail,
          /*ldi*/ headSize_even ? kStrideN : qk_gemm_K,
          /*ldo*/ kvTail,
          /*xtype*/
          vnni_pack ? XformTPP::XFORM_XPOSE_N2V_TPP : XformTPP::XFORM_XPOSE_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);
  // Create tpp transforms for Value
  auto v_xform = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ av_gemm_K,
          /*in_cols*/ headSize,
          /*out_rows*/ av_gemm_K,
          /*out_cols*/ headSize,
          /*ldi*/ av_gemm_K_even ? vStrideN : headSize,
          /*ldo*/ headSize,
          /*xtype*/ XformTPP::XFORM_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);
  auto v_xform_tail = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ av_gemm_K_tail,
          /*in_cols*/ headSize,
          /*out_rows*/ av_gemm_K_tail,
          /*out_cols*/ headSize,
          /*ldi*/ av_gemm_K_tail_even ? vStrideN : headSize,
          /*ldo*/ headSize,
          /*xtype*/ XformTPP::XFORM_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  // Reorder K, V
  at::parallel_for(
      0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, l = 0, n = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, l, kvSlice);
        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          n = l * kvSplitSize;
          auto psize = l * av_gemm_K;
          if (n + kvSplitSize < kvSize) {
            if (headSize_even) {
              // main
              k_xform(
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                  key_reorder_ptr + i * num_head * headSize * kvSize +
                      j * headSize * kvSize + n * headSize);
            } else if (!headSize_even && is_fp16) {
              // padding
              // [kvSplitSize, headSize] -> [kvSplitSize, headSize + 1]
              pad_col_zero(
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                  key_padding_ptr + i * num_head * qk_gemm_K * kvSize +
                      j * qk_gemm_K * kvSize + n * qk_gemm_K,
                  kvSplitSize,
                  headSize + 1,
                  kStrideN);
              k_xform(
                  key_padding_ptr + i * num_head * qk_gemm_K * kvSize +
                      j * qk_gemm_K * kvSize + n * qk_gemm_K,
                  key_reorder_ptr + i * num_head * qk_gemm_K * kvSize +
                      j * qk_gemm_K * kvSize + n * qk_gemm_K);
            }
            if (!av_gemm_K_even) {
              if (is_fp16 && vnni_pack) {
                // padding
                // [kvSplitSize, headSize] -> [kvSplitSize + 1, headSize]
                pad_row_zero(
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                    value_padding_ptr +
                        i * num_head * kv_padding_size * headSize +
                        j * kv_padding_size * headSize + psize * headSize,
                    av_gemm_K,
                    headSize,
                    vStrideN);
                v_xform(
                    value_padding_ptr +
                        i * num_head * kv_padding_size * headSize +
                        j * kv_padding_size * headSize + psize * headSize,
                    value_reorder_ptr +
                        i * num_head * kv_padding_size * headSize +
                        j * kv_padding_size * headSize + psize * headSize);
              }
            } else if (vnni_pack) {
              v_xform(
                  v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                  value_reorder_ptr +
                      i * num_head * kv_padding_size * headSize +
                      j * kv_padding_size * headSize + n * headSize);
            }
          } else {
            // tail
            if (headSize_even) {
              k_xform_tail(
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                  key_reorder_ptr + i * num_head * headSize * kvSize +
                      j * headSize * kvSize + n * headSize);
            } else if (!headSize_even && is_fp16) {
              // padding
              // [kvtail, headSize] -> [kvtail, headSize + 1]
              pad_col_zero(
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                  key_padding_ptr + i * num_head * qk_gemm_K * kvSize +
                      j * qk_gemm_K * kvSize + n * qk_gemm_K,
                  kvTail,
                  headSize + 1,
                  kStrideN);
              k_xform_tail(
                  key_padding_ptr + i * num_head * qk_gemm_K * kvSize +
                      j * qk_gemm_K * kvSize + n * qk_gemm_K,
                  key_reorder_ptr + i * num_head * qk_gemm_K * kvSize +
                      j * qk_gemm_K * kvSize + n * qk_gemm_K);
            }
            if (!av_gemm_K_tail_even) {
              if (is_fp16 && vnni_pack) {
                // padding
                // [kvtail, headSize] -> [kvtail + 1, headSize]
                pad_row_zero(
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                    value_padding_ptr +
                        i * num_head * kv_padding_size * headSize +
                        j * kv_padding_size * headSize + psize * headSize,
                    av_gemm_K_tail,
                    headSize,
                    vStrideN);
                v_xform_tail(
                    value_padding_ptr +
                        i * num_head * kv_padding_size * headSize +
                        j * kv_padding_size * headSize + psize * headSize,
                    value_reorder_ptr +
                        i * num_head * kv_padding_size * headSize +
                        j * kv_padding_size * headSize + psize * headSize);
              }
            } else if (vnni_pack) {
              v_xform_tail(
                  v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                  value_reorder_ptr +
                      i * num_head * kv_padding_size * headSize +
                      j * kv_padding_size * headSize + n * headSize);
            }
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
        }
      });

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;
        scalar_t* qk_reduced_data =
            buf_reduced_data + ompIdx * qSplitSize * av_gemm_K;
        scalar_t* query_t_padding_ptr = (is_fp16 && !headSize_even)
            ? query_padding_ptr + ompIdx * qSplitSize * qk_gemm_K
            : nullptr;

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          torch_ipex::cpu::kernel::fill_stub(
              qk_max_data,
              -std::numeric_limits<accum_t>::infinity(),
              qBlockSize);
          torch_ipex::cpu::kernel::fill_stub(
              qk_sum_data, static_cast<accum_t>(0), qBlockSize);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          if (is_fp16 && !headSize_even) {
            // pad query if headSize is not even for fp16
            // [qBlockSize, headSize] -> [qBlockSize, headSize + 1]
            pad_col_zero(
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                query_t_padding_ptr,
                qBlockSize,
                headSize + 1,
                qStrideM);
          }
          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            // Calculate scale * q @ k.T
            if ((!is_fp16 && headSize_even) || is_fp16) {
              if (qBlockSize == qSplitSize) {
                // q main
                if (n + kvSplitSize < kvSize) {
                  // k main
                  qk_gemm(
                      (is_fp16 && !headSize_even)
                          ? query_t_padding_ptr
                          : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                      key_reorder_ptr + i * num_head * qk_gemm_K * kvSize +
                          j * qk_gemm_K * kvSize + n * qk_gemm_K,
                      qk_data,
                      1);
                } else {
                  // k tail
                  qk_gemm_ktail(
                      (is_fp16 && !headSize_even)
                          ? query_t_padding_ptr
                          : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                      key_reorder_ptr + i * num_head * qk_gemm_K * kvSize +
                          j * qk_gemm_K * kvSize + n * qk_gemm_K,
                      qk_data,
                      1);
                }
              } else {
                if (n + kvSplitSize < kvSize) {
                  // k main
                  qk_gemm_qtail(
                      (is_fp16 && !headSize_even)
                          ? query_t_padding_ptr
                          : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                      key_reorder_ptr + i * num_head * qk_gemm_K * kvSize +
                          j * qk_gemm_K * kvSize + n * qk_gemm_K,
                      qk_data,
                      1);
                } else {
                  // k tail
                  qk_gemm_qktail(
                      (is_fp16 && !headSize_even)
                          ? query_t_padding_ptr
                          : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                      key_reorder_ptr + i * num_head * qk_gemm_K * kvSize +
                          j * qk_gemm_K * kvSize + n * qk_gemm_K,
                      qk_data,
                      1);
                }
              }
            } else {
              _mkl_gemm(
                  CblasColMajor,
                  CblasTrans,
                  CblasNoTrans,
                  kvBlockSize,
                  qBlockSize,
                  headSize,
                  static_cast<accum_t>(1),
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                  kStrideN,
                  q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                  qStrideM,
                  static_cast<accum_t>(0),
                  qk_data,
                  kvBlockSize);
            }
            // Apply causal mask, fill unused with -inf
            if (is_causal && num_keys - n <= kvSplitSize) {
              for (const auto row : c10::irange(qBlockSize)) {
                int64_t last_col = m + row - n;
                accum_t* row_ptr = qk_data + row * kvBlockSize;
                torch_ipex::cpu::kernel::fill_stub(
                    row_ptr + last_col + 1,
                    -std::numeric_limits<accum_t>::infinity(),
                    kvBlockSize - last_col - 1);
              }
            }
            // Update attention weights with attention mask
            // And apply scaling factor
            if (attention_mask.has_value()) {
              for (int64_t row = 0; row < qBlockSize; ++row) {
                // qk <- qk * scaling_factor + attn_mask, else
                if (mStrideN == 0) {
                  _scale_attn_mask_fusion_kernel</*is_stride_zero*/ true>(
                      qk_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM,
                      kvBlockSize,
                      qk_data + row * kvBlockSize,
                      scaling_factor);
                } else {
                  _scale_attn_mask_fusion_kernel</*is_stride_zero*/ false>(
                      qk_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM + n,
                      kvBlockSize,
                      qk_data + row * kvBlockSize,
                      scaling_factor);
                }
              }
            }
            // Update coefficients with Softmax
            accum_t tmp_max = 0, tmp_sum = 0, sum_old = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
              sum_old = qk_sum_data[row];
              if (attention_mask.has_value()) {
                // max per row
                tmp_max = at::vec::reduce_all<accum_t>(
                    [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
                    qk_data + row * kvBlockSize,
                    kvBlockSize);
              } else {
                // apply scaling factor and max per row in fusion
                _mul_reduce_max_fusion_kernel(
                    qk_data + row * kvBlockSize,
                    scaling_factor,
                    kvBlockSize,
                    qk_data + row * kvBlockSize,
                    tmp_max);
              }
              tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
              // qk <- exp(qk - max) and sum per row
              tmp_sum = tmp_max;
              _exp_reduce_sum_fusion_kernel(
                  qk_data + row * kvBlockSize,
                  kvBlockSize,
                  qk_reduced_data +
                      row *
                          ((kvBlockSize % 2) != 0 ? 1 + kvBlockSize
                                                  : kvBlockSize),
                  tmp_sum);
              // exp_tmp <- exp(max[row] - max)
              exp_tmp = std::exp(qk_max_data[row] - tmp_max);
              // sum[row] <- sum + exp_tmp * sum[row]
              qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
              // max[row] <- max
              qk_max_data[row] = tmp_max;
              // dst <- dst * exp_tmp
              if (n > 0) {
                at::vec::map<accum_t>(
                    [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                    dst_data + row * headSize,
                    dst_data + row * headSize,
                    headSize);
              }
              // Zero padding: [qSplitSize,kvSplitSize] ->
              // [qSplitSize,kvSplitSize + 1]
              if (kvBlockSize % 2 != 0) {
                *(qk_reduced_data + row * (1 + kvBlockSize) + kvBlockSize) =
                    scalar_t(0);
              }
            }

            // Calculate Softmax(q @ k.T) @ v
            if (((!is_fp16 && av_gemm_K_even && av_gemm_K_tail_even) ||
                 is_fp16)) {
              int64_t psize = n / kvSplitSize * av_gemm_K;
              if (n + kvSplitSize < kvSize) {
                // main
                if (n == 0) {
                  av_gemm(
                      qk_reduced_data,
                      vnni_pack
                          ? (value_reorder_ptr +
                             i * num_head * kv_padding_size * headSize +
                             j * kv_padding_size * headSize + psize * headSize)
                          : (v_data + i * vStrideB + j * vStrideH +
                             n * vStrideN),
                      dst_data,
                      1);
                } else {
                  // bias
                  av_gemm_bias(
                      qk_reduced_data,
                      vnni_pack
                          ? (value_reorder_ptr +
                             i * num_head * kv_padding_size * headSize +
                             j * kv_padding_size * headSize + psize * headSize)
                          : (v_data + i * vStrideB + j * vStrideH +
                             n * vStrideN),
                      dst_data,
                      1);
                }
              } else if (n + kvSplitSize >= kvSize) {
                // tail
                if (n == 0) {
                  av_gemm_tail(
                      qk_reduced_data,
                      vnni_pack
                          ? (value_reorder_ptr +
                             i * num_head * kv_padding_size * headSize +
                             j * kv_padding_size * headSize + psize * headSize)
                          : (v_data + i * vStrideB + j * vStrideH +
                             n * vStrideN),
                      dst_data,
                      1);
                } else {
                  // bias
                  av_gemm_bias_tail(
                      qk_reduced_data,
                      vnni_pack
                          ? (value_reorder_ptr +
                             i * num_head * kv_padding_size * headSize +
                             j * kv_padding_size * headSize + psize * headSize)
                          : (v_data + i * vStrideB + j * vStrideH +
                             n * vStrideN),
                      dst_data,
                      1);
                }
              }
            } else {
              _mkl_gemm(
                  CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  headSize,
                  qBlockSize,
                  kvBlockSize,
                  static_cast<accum_t>(1),
                  v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                  vStrideN,
                  qk_reduced_data,
                  kvBlockSize % 2 == 0 ? kvBlockSize : kvBlockSize + 1,
                  n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
                  dst_data,
                  headSize);
            }
          }
          // dst <- dst / sum[row]
          // reorder MHA output with strides
          for (int64_t row = 0; row < qBlockSize; ++row) {
            accum_t sum_reciprocal = 1 / qk_sum_data[row];
            at::vec::map<scalar_t>(
                [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
                out_data + i * oStrideB + j * oStrideH + m * oStrideM +
                    row * oStrideM,
                dst_data + row * headSize,
                headSize);
          }
          // Store logsumexp for backward
          accum_t* lse_ptr =
              lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
          for (const auto row : c10::irange(qBlockSize)) {
            lse_ptr[row * lStrideM] =
                qk_max_data[row] + std::log(qk_sum_data[row]);
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
}

#define AT_DISPATCH_MASK_TYPES(TYPE, NAME, ...)                      \
  AT_DISPATCH_SWITCH(                                                \
      TYPE,                                                          \
      NAME,                                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                               \
          at::ScalarType::Bool, mask_t, __VA_ARGS__)                 \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                           \
              at::ScalarType::Float, mask_t, __VA_ARGS__)            \
              AT_PRIVATE_CASE_TYPE_USING_HINT(                       \
                  at::ScalarType::Double, mask_t, __VA_ARGS__)       \
                  AT_PRIVATE_CASE_TYPE_USING_HINT(                   \
                      at::ScalarType::BFloat16, mask_t, __VA_ARGS__) \
                      AT_PRIVATE_CASE_TYPE_USING_HINT(               \
                          at::ScalarType::Half, mask_t, __VA_ARGS__))

void flash_attention_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  auto q_seq_len = query.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, query.scalar_type(), "flash_attention", [&] {
        if (!attention_mask.has_value()) {
          if (q_seq_len >= 768) {
            cpu_flash_attention<scalar_t, scalar_t, 256, 512>(
                output,
                logsumexp,
                query,
                key,
                value,
                dropout_p,
                is_causal,
                attention_mask,
                scale);
          } else if (q_seq_len >= 192) {
            cpu_flash_attention<scalar_t, scalar_t, 64, 512>(
                output,
                logsumexp,
                query,
                key,
                value,
                dropout_p,
                is_causal,
                attention_mask,
                scale);
          } else {
            cpu_flash_attention<scalar_t, scalar_t, 32, 512>(
                output,
                logsumexp,
                query,
                key,
                value,
                dropout_p,
                is_causal,
                attention_mask,
                scale);
          }
        } else {
          AT_DISPATCH_MASK_TYPES(
              attention_mask.value().scalar_type(),
              "flash_attention_mask",
              [&]() {
                if (q_seq_len >= 768) {
                  cpu_flash_attention<scalar_t, mask_t, 256, 512>(
                      output,
                      logsumexp,
                      query,
                      key,
                      value,
                      dropout_p,
                      is_causal,
                      attention_mask,
                      scale);
                } else if (q_seq_len >= 192) {
                  cpu_flash_attention<scalar_t, mask_t, 64, 512>(
                      output,
                      logsumexp,
                      query,
                      key,
                      value,
                      dropout_p,
                      is_causal,
                      attention_mask,
                      scale);
                } else {
                  cpu_flash_attention<scalar_t, mask_t, 32, 512>(
                      output,
                      logsumexp,
                      query,
                      key,
                      value,
                      dropout_p,
                      is_causal,
                      attention_mask,
                      scale);
                }
              });
        }
      });
}

std::tuple<at::Tensor, at::Tensor> flash_attention_kernel(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  RECORD_FUNCTION(
      "torch_ipex::flash_attention_kernel", c10::ArrayRef<c10::IValue>({}));

  const auto dtype = query.scalar_type();
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(3);

  TORCH_CHECK(
      c10::isFloatingType(dtype),
      "IPEX flash_attention: Expected data type in FP32, FP64, BF16, FP16, but got ",
      dtype,
      " instead.");
  TORCH_CHECK(
      dtype == key.scalar_type() && dtype == value.scalar_type(),
      "IPEX flash_attention: Q/K/V should have the same data type");
  TORCH_CHECK(
      !attention_mask.has_value() ||
          dtype == attention_mask.value().scalar_type() ||
          attention_mask.value().scalar_type() == ScalarType::Float ||
          attention_mask.value().scalar_type() == ScalarType::Bool,
      "IPEX flash_attention: Mask should have the same data type as Q/K/V or Bool");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "IPEX flash_attention: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(
      dropout_p == 0.0,
      "IPEX flash_attention: Currently do not support dropout > 0");
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "IPEX flash_attention: Q/K/V should have the same head size");
  TORCH_CHECK(
      (query.stride(-1) == 1) && (key.stride(-1) == 1) &&
          (value.stride(-1) == 1) &&
          (!attention_mask.has_value() ||
           attention_mask.value().stride(-1) == 1),
      "IPEX flash_attention: Q/K/V/Mask should be continuous on the last dim");

  std::optional<at::Tensor> attn_mask =
      convert_boolean_attn_mask(attention_mask, query.dtype());
  at::Tensor output =
      at::empty({batchSize, qSize, num_head, headSize}, query.options());
  const auto accumulate_dtype = at::toOpMathType(dtype);
  at::Tensor logsumexp = at::empty(
      {batchSize, qSize, num_head}, query.options().dtype(accumulate_dtype));

  flash_attention_kernel_impl(
      output,
      logsumexp,
      query,
      key,
      value,
      dropout_p,
      is_causal,
      attn_mask,
      scale);

  output = output.transpose(1, 2);
  logsumexp = logsumexp.transpose(1, 2);

  return std::make_tuple(std::move(output), std::move(logsumexp));
}
} // anonymous namespace

IPEX_REGISTER_DISPATCH(flash_attention_kernel_stub, &flash_attention_kernel);

} // namespace cpu
} // namespace torch_ipex
