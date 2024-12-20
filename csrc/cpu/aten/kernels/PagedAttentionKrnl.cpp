#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <aten/PagedAttention.h>
#include <aten/utils/mkl_gemm.h>
#include <c10/util/irange.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <omp.h>
#include <limits>
#include "vec/vec.h"

#define PARTITION_SIZE 128

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

namespace torch_ipex {
namespace cpu {

namespace {

namespace fp8 {

template <typename DST_T, typename SRC_T>
void scaled_convert(
    const SRC_T* src_ptr,
    DST_T* dst_ptr,
    size_t len,
    float scale) {
  torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(dst_ptr, src_ptr, len);
}

template <>
void scaled_convert<at::Float8_e5m2, float>(
    const float* src_ptr,
    at::Float8_e5m2* dst_ptr,
    size_t len,
    float scale) {
  size_t idx_offset = 0;
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::cvt_fp32_e5m2_rne_intrinsic(src_ptr, dst_ptr, len);
#else
  for (size_t i = 0; i < len; i++) {
    dst_ptr[i] = static_cast<at::Float8_e5m2>(src_ptr[i]);
  }
#endif
}

template <>
void scaled_convert<at::Float8_e5m2, at::BFloat16>(
    const at::BFloat16* src_ptr,
    at::Float8_e5m2* dst_ptr,
    size_t len,
    float scale) {
  size_t idx_offset = 0;
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::cvt_bf16_e5m2_rne_intrinsic(src_ptr, dst_ptr, len);
#else
  for (size_t i = 0; i < len; i++) {
    dst_ptr[i] = static_cast<at::Float8_e5m2>(src_ptr[i]);
  }
#endif
}

} // namespace fp8

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

template <typename QT, typename KT>
void reduce_head(
    const QT* q_ptr_start,
    int64_t kv_head_group_size,
    const KT* k_cache_start,
    float* attn_w_pos,
    int attn_w_stride,
    int64_t head_size) {
#if defined(CPU_CAPABILITY_AVX512)
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    torch_ipex::cpu::kernel::_reduce_head<QT, KT, KT>(
        q_ptr_start + i * head_size,
        k_cache_start,
        attn_w_pos + i * attn_w_stride,
        head_size,
        false,
        nullptr);
  }
#else
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    for (auto hsi = 0; hsi < head_size; hsi++) {
      attn_w_pos[i * attn_w_stride] +=
          (float)q_ptr_start[i * head_size + hsi] * (float)k_cache_start[hsi];
    }
  }

#endif
}

#if defined(CPU_CAPABILITY_AVX512)
void _reduce_head_e5m2(
    const at::BFloat16* q_ptr_start,
    const at::Float8_e5m2* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size) {
  using namespace torch_ipex::cpu::kernel;
  auto hsi = 0;
  auto vec_size = 32;
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto k_vec_ =
        _mm512_cvte5m2_fp16(_mm256_loadu_si256((__m256i*)&k_ptr_start[hsi]));
    auto k_vec0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(k_vec_, 0));
    auto k_vec1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(k_vec_, 1));
    auto q_vec0 = _loadu(q_ptr_start + hsi);
    auto q_vec1 = _loadu(q_ptr_start + hsi + 16);
    qk_sum_vec = _mm512_fmadd_ps(q_vec0, k_vec0, qk_sum_vec);
    qk_sum_vec = _mm512_fmadd_ps(q_vec1, k_vec1, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    attn_w_pos[0] += q_ptr_start[hsi] * (float)k_ptr_start[hsi];
  }
}

inline void _mul_and_accumulate_e5m2(
    const float& attn_w,
    const at::Float8_e5m2* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    int accumulated) {
  using namespace torch_ipex::cpu::kernel;
  auto hsi = 0;
  auto vec_size = 32;
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto attn_w_vec = _mm512_set1_ps(attn_w);
    auto v_vec_ =
        _mm512_cvte5m2_fp16(_mm256_loadu_si256((__m256i*)&v_ptr_start[hsi]));
    auto v_vec0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v_vec_, 0));
    auto v_vec1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v_vec_, 1));
    if (accumulated) {
      auto attn_out_vec0 = _loadu(attn_out_start + hsi);
      auto attn_out_vec1 = _loadu(attn_out_start + hsi + 16);
      auto attn_out_vec_new0 =
          _mm512_fmadd_ps(attn_w_vec, v_vec0, attn_out_vec0);
      auto attn_out_vec_new1 =
          _mm512_fmadd_ps(attn_w_vec, v_vec1, attn_out_vec1);
      _storeu(attn_out_start + hsi, attn_out_vec_new0);
      _storeu(attn_out_start + hsi + 16, attn_out_vec_new1);
    } else {
      auto attn_out_vec_new0 = _mm512_mul_ps(attn_w_vec, v_vec0);
      auto attn_out_vec_new1 = _mm512_mul_ps(attn_w_vec, v_vec1);
      _storeu(attn_out_start + hsi, attn_out_vec_new0);
      _storeu(attn_out_start + hsi + 16, attn_out_vec_new1);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulated) {
      attn_out_start[hsi] += attn_w * (float)v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * (float)v_ptr_start[hsi];
    }
  }
}
#endif

template <>
void reduce_head<at::BFloat16, at::Float8_e5m2>(
    const at::BFloat16* q_ptr_start,
    int64_t kv_head_group_size,
    const at::Float8_e5m2* k_cache_start,
    float* attn_w_pos,
    int attn_w_stride,
    int64_t head_size) {
#if defined(CPU_CAPABILITY_AVX512)
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    _reduce_head_e5m2(
        q_ptr_start + i * head_size,
        k_cache_start,
        attn_w_pos + i * attn_w_stride,
        head_size);
  }
#else
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    for (auto hsi = 0; hsi < head_size; hsi++) {
      attn_w_pos[i * attn_w_stride] +=
          (float)q_ptr_start[i * head_size + hsi] * (float)k_cache_start[hsi];
    }
  }

#endif
}

template <typename OT, typename CT>
inline void mul_attenion_weights_and_value_of_head(
    const float* attn_w,
    int attn_w_stride,
    const CT* v_cache_start,
    OT* attn_out_start,
    int attn_out_strideH,
    int kv_head_group_size,
    int64_t head_size,
    bool accumulated) {
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512)
  for (auto i = 0; i < kv_head_group_size; i++) {
    torch_ipex::cpu::kernel::_mul_and_accumulate<CT, OT, CT>(
        attn_w[i * attn_w_stride],
        v_cache_start,
        attn_out_start + i * attn_out_strideH,
        head_size,
        false,
        nullptr,
        accumulated);
  }
#else
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (hsi = 0; hsi < head_size; hsi++) {
      if (accumulated) {
        attn_out_start[i * attn_out_strideH + hsi] +=
            attn_w[i * attn_w_stride] * (float)v_cache_start[hsi];
      } else {
        attn_out_start[i * attn_out_strideH + hsi] =
            attn_w[i * attn_w_stride] * (float)v_cache_start[hsi];
      }
    }
  }
#endif
}

template <>
inline void mul_attenion_weights_and_value_of_head<float, at::Float8_e5m2>(
    const float* attn_w,
    int attn_w_stride,
    const at::Float8_e5m2* v_cache_start,
    float* attn_out_start,
    int attn_out_strideH,
    int kv_head_group_size,
    int64_t head_size,
    bool accumulated) {
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512)
  for (auto i = 0; i < kv_head_group_size; i++) {
    _mul_and_accumulate_e5m2(
        attn_w[i * attn_w_stride],
        v_cache_start,
        attn_out_start + i * attn_out_strideH,
        head_size,
        accumulated);
  }
#else
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (hsi = 0; hsi < head_size; hsi++) {
      if (accumulated) {
        attn_out_start[i * attn_out_strideH + hsi] +=
            attn_w[i * attn_w_stride] * (float)v_cache_start[hsi];
      } else {
        attn_out_start[i * attn_out_strideH + hsi] =
            attn_w[i * attn_w_stride] * (float)v_cache_start[hsi];
      }
    }
  }
#endif
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
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    at::native::_store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale + alibi_mask
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_alibi_reduce_max_fusion_kernel(
    scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max,
    const int& token_start,
    const int& context_len,
    const scalar_t& alibi_slope) {
  for (auto i = 0; i < size; i++) {
    a[i] = a[i] * scale;
    auto alibi_slopes_val = alibi_slope * (i + token_start + 1 - context_len);
    a[i] += alibi_slopes_val;
    max = std::max(max, a[i]);
  }
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    tmp1.store(out + i);
  }
  tmp_max = at::vec::vec_reduce_all<scalar_t>(
      [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
        return at::vec::maximum(x, y);
      },
      vec_tmp_max);
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = tmp_max;
}

/**
 * Performs scale-dot-product for the next token based on cached key-value
 * attention.
 *
 * This function computes the attention weights and applies the attention
 * mechanism to obtain the final output. It takes in tensors representing the
 * query, key cache, value cache, head mapping, scale, block tables, context
 * lengths, block size, max context length, and optional alibi slopes. The
 * output tensor is updated with the computed attention values.
 *
 * @param out           Output tensor [num_seqs, num_heads, head_size].
 * @param query         Query tensor [num_seqs, num_heads, head_size].
 * @param key_cache     The pre-allocated buffer to store the key cache. The
 * shape should be [num_blocks, block_size, num_heads, head_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, block_size, num_heads, head_size].
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [num_seqs].
 * @param block_size    The block size which means the number of token in every
 * block.
 * @param max_context_len Maximum context length.
 * @param alibi_slopes  Optional tensor of alibi slopes with the shape of
 * (num_heads).
 * @param k_scale       Scaling factor for key cache of data type fp8.
 * @param v_scale       Scaling factor for value cache of data type fp8.
 */
template <typename scalar_t, typename cache_t>
void single_query_cached_kv_attention_kernel(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const double scale,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes,
    const double k_scale,
    const double v_scale) {
  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<cache_t>();
  auto value_cache_ptr = value_cache.data_ptr<cache_t>();
  auto block_tables_ptr = block_tables.data_ptr<int>();
  auto context_lens_ptr = context_lens.data_ptr<int>();
  auto alibi_slopes_ptr = alibi_slopes.has_value()
      ? alibi_slopes.value().data_ptr<float>()
      : nullptr;
  auto num_seqs = query.size(0);
  auto num_heads = query.size(1);
  auto head_size = query.size(2);
  auto num_kv_heads = key_cache.size(1);
  auto kv_head_group_size = num_heads / num_kv_heads;
  auto max_num_blocks_per_seq = block_tables.size(1);

  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = key_cache.stride(2);
  auto kv_block_strideH = key_cache.stride(1);

  auto out_strideN = out.stride(0);
  auto out_strideH = out.stride(1);

  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);

  auto max_num_partitions =
      (max_context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

  auto max_logits = at::empty(
      {num_seqs, num_heads, max_num_partitions + 1},
      query.options().dtype(at::ScalarType::Float));

  auto exp_sum = at::empty(
      {num_seqs, num_heads, max_num_partitions + 1},
      query.options().dtype(at::ScalarType::Float));

  auto tmp_out = at::empty(
      {num_seqs, num_heads, max_num_partitions, head_size},
      query.options().dtype(at::ScalarType::Float));

  auto tmp_out_ptr = tmp_out.data_ptr<float>();
  auto max_logits_ptr = max_logits.data_ptr<float>();
  auto exp_sum_ptr = exp_sum.data_ptr<float>();

  auto max_logits_strideN = max_logits.stride(0);
  auto max_logits_strideH = max_logits.stride(1);
  auto exp_sum_strideN = exp_sum.stride(0);
  auto exp_sum_strideH = exp_sum.stride(1);
  auto tmp_out_strideN = tmp_out.stride(0);
  auto tmp_out_strideH = tmp_out.stride(1);
  auto tmp_out_strideS = tmp_out.stride(2);

  auto max_logic_blocks = (max_context_len + block_size - 1) / block_size;

  auto thread_numbers = omp_get_max_threads();
  auto max_parallel_parts = thread_numbers * 4;

  auto tmp_logits = at::empty(
      {thread_numbers, kv_head_group_size, PARTITION_SIZE},
      query.options().dtype(at::ScalarType::Float));
  auto logits_ptrs = tmp_logits.data_ptr<float>();

  if (alibi_slopes.has_value()) {
    auto alibi_slopes_size = alibi_slopes.value().size(0);
    TORCH_CHECK(
        alibi_slopes_size == num_heads,
        "alibi_slopes size is not equal to num_heads");
  }
#pragma omp parallel for collapse(3) schedule(static, 1)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto partition_id = 0; partition_id < max_num_partitions;
         partition_id++) {
      for (auto head_group_start = 0; head_group_start < num_heads;
           head_group_start += kv_head_group_size) {
        auto omp_thread_id = omp_get_thread_num();
        auto context_len = context_lens_ptr[seq_id];
        auto partition_start = partition_id * PARTITION_SIZE;
        if (partition_start >= context_len)
          continue;
        auto partition_end =
            std::min(partition_start + PARTITION_SIZE, context_len);
        auto token_num = partition_end - partition_start;
        auto block_num = (token_num + block_size - 1) / block_size;
        auto logical_block_start = partition_start / block_size;
        auto logical_block_end = logical_block_start + block_num;
        auto kv_head_id = head_group_start / kv_head_group_size;
        auto q_ptr_start =
            query_ptr + seq_id * q_strideN + head_group_start * q_strideH;
        auto max_logits_offset = seq_id * max_logits_strideN +
            head_group_start * max_logits_strideH + partition_id;
        auto exp_sum_offset = seq_id * exp_sum_strideN +
            head_group_start * exp_sum_strideH + partition_id;
        //{num_seqs, num_heads, max_num_partitions, head_size}
        auto tmp_out_start = tmp_out_ptr + seq_id * tmp_out_strideN +
            head_group_start * tmp_out_strideH + partition_id * tmp_out_strideS;
        float* logits =
            logits_ptrs + omp_thread_id * PARTITION_SIZE * kv_head_group_size;
        auto logits_position = 0;
        // 1)calculate the matmul(query, key) for this partition
        for (auto logical_block_id = logical_block_start;
             logical_block_id < logical_block_end;
             logical_block_id++) {
          auto physical_block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + logical_block_id];
          auto tokens_in_block =
              std::min(block_size, context_len - logical_block_id * block_size);
          auto token_start = logical_block_id * block_size;
          auto token_end = token_start + tokens_in_block;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto block_offset = token_id - token_start;
            auto k_cache_start = key_cache_ptr +
                physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP + kv_head_id * kv_block_strideH;
            reduce_head(
                q_ptr_start,
                kv_head_group_size,
                k_cache_start,
                &(logits[logits_position]),
                PARTITION_SIZE,
                head_size);
            logits_position++;
          }
        }
        // 2) calculate the max and exp_sum for this partition
        for (int hi = 0; hi < kv_head_group_size; hi++) {
          auto partition_max = -std::numeric_limits<float>::infinity();
          if (alibi_slopes_ptr != nullptr) {
            _mul_alibi_reduce_max_fusion_kernel<float>(
                logits + hi * PARTITION_SIZE,
                scale,
                token_num,
                logits + hi * PARTITION_SIZE,
                partition_max,
                partition_start,
                context_len,
                alibi_slopes_ptr[head_group_start + hi]);
          } else {
            _mul_reduce_max_fusion_kernel<float>(
                logits + hi * PARTITION_SIZE,
                scale,
                token_num,
                logits + hi * PARTITION_SIZE,
                partition_max);
          }
          max_logits_ptr[max_logits_offset + hi * max_logits_strideH] =
              partition_max;
          _exp_reduce_sum_fusion_kernel<float, float>(
              logits + hi * PARTITION_SIZE,
              token_num,
              logits + hi * PARTITION_SIZE,
              partition_max);
          exp_sum_ptr[exp_sum_offset + hi * exp_sum_strideH] = partition_max;
        }

        // 3) calculate the matmul(exp(logits-partition_max), value) for this
        // partition, need to divide the global exp_sum in the final result.
        logits_position = 0;
        for (auto logical_block_id = logical_block_start;
             logical_block_id < logical_block_end;
             logical_block_id++) {
          auto physical_block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + logical_block_id];
          auto tokens_in_block =
              std::min(block_size, context_len - logical_block_id * block_size);
          auto token_start = logical_block_id * block_size;
          auto token_end = token_start + tokens_in_block;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto block_offset = token_id - token_start;
            auto v_cache_start = value_cache_ptr +
                physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP + kv_head_id * kv_block_strideH;
            auto accumulated = logits_position > 0;
            mul_attenion_weights_and_value_of_head(
                &(logits[logits_position]),
                PARTITION_SIZE,
                v_cache_start,
                tmp_out_start,
                tmp_out_strideH,
                kv_head_group_size,
                head_size,
                accumulated);
            logits_position++;
          }
        }
      }
    }
  }

// calculate the final output
#pragma omp parallel for collapse(2)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto global_max = -std::numeric_limits<float>::infinity();
      auto global_exp_sum = 0.0;
      auto context_len = context_lens_ptr[seq_id];
      auto partition_num = (context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
      // calculate the global max and exp_sum for this head
      for (auto partition_id = 0; partition_id < max_num_partitions;
           partition_id++) {
        if (partition_id >= partition_num)
          break;
        auto max_logit = max_logits_ptr
            [seq_id * max_logits_strideN + head_id * max_logits_strideH +
             partition_id];
        global_max = std::max(global_max, max_logit);
      }
      // update the partition 0 result with the global max
      auto partition0_out_start =
          tmp_out_ptr + seq_id * tmp_out_strideN + head_id * tmp_out_strideH;
      auto max_logit0 = max_logits_ptr
          [seq_id * max_logits_strideN + head_id * max_logits_strideH];
      float exp_val = expf(max_logit0 - global_max);
      global_exp_sum +=
          exp_sum_ptr[seq_id * exp_sum_strideN + head_id * exp_sum_strideH] *
          exp_val;
      at::vec::Vectorized<float> exp_val_vec0(exp_val);
      at::vec::map<float>(
          [&](auto a) { return a * exp_val_vec0; },
          partition0_out_start,
          partition0_out_start,
          head_size);

      // accumulate the partition 1 to partition n result into partition 0
      if (partition_num > 1) {
        for (auto partition_id = 1; partition_id < partition_num;
             partition_id++) {
          if (partition_id * PARTITION_SIZE >= context_len)
            break;
          auto tmp_out_start = tmp_out_ptr + seq_id * tmp_out_strideN +
              head_id * tmp_out_strideH + partition_id * tmp_out_strideS;
          auto max_logit = max_logits_ptr
              [seq_id * max_logits_strideN + head_id * max_logits_strideH +
               partition_id];
          auto exp_sum = exp_sum_ptr
              [seq_id * exp_sum_strideN + head_id * exp_sum_strideH +
               partition_id];
          exp_val = expf(max_logit - global_max);
          global_exp_sum += exp_sum * exp_val;
          at::vec::Vectorized<float> exp_val_vec(exp_val);
          at::vec::map2<float>(
              [&](auto a, auto b) { return a + exp_val_vec * b; },
              partition0_out_start,
              partition0_out_start,
              tmp_out_start,
              head_size);
        }
      }

      // copy the partition 0 result into attn_outs
      auto attn_out_start =
          out_ptr + seq_id * out_strideN + head_id * out_strideH;
      float inverse_global_sum = 1.0 / (global_exp_sum + 1e-8);
      at::vec::Vectorized<float> inverse_global_sum_vec(inverse_global_sum);
      // rescale the partition 0 result with global exp_sum
      at::vec::map<float>(
          [&](auto a) { return a * inverse_global_sum_vec; },
          partition0_out_start,
          partition0_out_start,
          head_size);
      // copy the partition 0 result into attn_outs
      at::vec::map<scalar_t>(
          [&](auto a) { return a; },
          attn_out_start,
          partition0_out_start,
          head_size);
    }
  }
} // single_query_cached_kv_attention_kernel

/**
 * Reshapes and caches the key and value tensors based on the provided slot
 * mapping.
 *
 * @param key The input key tensor. The shape should be [num_seqs, num_heads,
 * head_size].
 * @param value The input value tensor.  The shape should be [num_seqs,
 * num_heads, head_size].
 * @param key_cache The output key cache tensor. The pre-allocated buffer to
 * store the key cache. The shape should be [num_blocks, block_size, num_heads,
 * head_size].
 * @param value_cache The output value cache tensor. The pre-allocated buffer to
 * store the value cache. The shape should be [num_blocks, block_size,
 * num_heads, head_size].
 * @param slot_mapping The slot mapping tensor. It stores the position to store
 * the key/value in the pre-allocated buffers. The shape should be the number of
 * sequences. For sequence i, the slot_mapping[i]//block_number can get the
 * block index, and the slot_mapping%block_size can get the offset of this
 * block.
 * @param k_scale Scaling factor for key cache of data type fp8.
 * @param v_scale Scaling factor for value cache of data type fp8.
 *
 * @tparam DST_T The data type of the output tensors.
 * @tparam SRC_T The data type of the input tensors.
 */
template <typename DST_T, typename SRC_T>
void reshape_and_cache_kernel(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping,
    const double k_scale,
    const double v_scale) {
  auto num_tokens = key.size(0);
  auto head_num = key.size(1);
  auto head_size = key.size(2);
  auto block_size = key_cache.size(2);
  auto hidden_size = head_num * head_size;
  auto key_cache_ptr = key_cache.data_ptr<DST_T>();
  auto key_ptr = key.data_ptr<SRC_T>();
  auto value_cache_ptr = value_cache.data_ptr<DST_T>();
  auto value_ptr = value.data_ptr<SRC_T>();
  auto slot_mapping_ptr = slot_mapping.data_ptr<int>();
  auto cache_strideN = key_cache.stride(0);
  auto cache_strideP = key_cache.stride(2);
  auto cache_strideH = key_cache.stride(1);
  auto key_state_strideN = key.stride(0);
  auto key_state_strideH = key.stride(1);
  auto value_state_strideN = value.stride(0);
  auto value_state_strideH = value.stride(1);
#pragma omp parallel for collapse(2)
  for (auto ti = 0; ti < num_tokens; ti++) {
    for (auto hi = 0; hi < head_num; hi++) {
      auto physical_block_id = slot_mapping_ptr[ti] / block_size;
      auto block_offset = slot_mapping_ptr[ti] % block_size;
      auto cache_offset = physical_block_id * cache_strideN +
          block_offset * cache_strideP + hi * cache_strideH;
      auto key_state_offset = ti * key_state_strideN + hi * key_state_strideH;
      auto value_state_offset =
          ti * value_state_strideN + hi * value_state_strideH;
      auto key_cache_start = key_cache_ptr + cache_offset;
      auto key_ptr_start = key_ptr + key_state_offset;
      auto value_cache_start = value_cache_ptr + cache_offset;
      auto value_ptr_start = value_ptr + value_state_offset;
      fp8::scaled_convert<DST_T, SRC_T>(
          key_ptr_start, key_cache_start, head_size, k_scale);
      fp8::scaled_convert<DST_T, SRC_T>(
          value_ptr_start, value_cache_start, head_size, v_scale);
    }
  }
}

/**
 * Performs scale-dot-product for the chunked prefill case.
 * In this case, we assume the current key/value already been cached.
 * The attention weights are calculated based on the query and the cached
 * key/value. If causal=True, the causal mask is aligned to the bottom right
 * corner of the attention matrix. For example, if seqlen_q = 2 and seqlen_k =
 * 5, the causal mask (1 = keep, 0 = masked out) is: 1 1 1 1 0 1 1 1 1 1 if
 * seqlen_q = 5 and seqlen_k = 2, the causal mask is: 0 0 0 0 0 0 1 0 1 1 If the
 * row of the mask is all zero, the output will be zero.
 *
 * For the chuned prefill case, the data layout is as follow:
 *
 * Definition of context_len, query_len, and seq_len.
 *   |---------- N-1 iteration --------|
 *   |---------------- N iteration ---------------------|
 *   |- tokenA -|......................|-- newTokens ---|
 *   |---------- context_len ----------|
 *   |-------------------- seq_len ---------------------|
 *                                     |-- query_len ---|
 *
 * when chunked prefill is enabled, prefill tokens and decode tokens can be
 * batched together in a flattened 1D query.
 *  |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
 *  |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|
 * For the flash_attn_varlen kernel, only chunked prefill tokens are processed
 * by this kernel. The decode tokens are processed by the
 * single_query_cached_kv_attention_kernel.
 */
template <typename scalar_t, typename cache_t, int64_t q_split_size = 16>
void flash_attn_varlen_kernel(
    at::Tensor& out, // [num_seqs, num_heads, head_size]
    at::Tensor& query, // [num_seqs, num_heads, head_size]
    at::Tensor& key_cache, // [num_blocks, num_heads, block_size,  head_size]
    at::Tensor& value_cache, //[num_blocks, num_heads, block_size, head_size]
    at::Tensor& cu_seqlens_q, // [batch_size+1] // the accumulted sequence
                              // length of query
    at::Tensor& cu_seqlens_k, // [batch_size+1] // the accumulted sequence
                              // length of key(cached)
    int64_t max_seqlen_q, // max sequence length of query
    int64_t max_seqlens_k, // max sequence length of key and value(cached,
                           // past+current)
    const double softmax_scale, // scale for softmax
    bool is_causal, // whether the attention is causal
    at::Tensor& block_table,
    const c10::optional<at::Tensor>& alibi_slopes,
    const double k_scale,
    const double v_scale) {
  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideH = key_cache.stride(1);
  auto kv_block_strideP = key_cache.stride(2);
  auto out_strideN = out.stride(0);
  auto out_strideH = out.stride(1);
  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);

  auto num_heads = query.size(1);
  auto head_size = query.size(2);
  auto num_kv_heads = key_cache.size(1);
  auto kv_head_group_size = num_heads / num_kv_heads;
  auto max_num_blocks_per_seq = block_table.size(1);
  auto batch_size = cu_seqlens_q.size(0) - 1;
  auto block_size = key_cache.size(2);

  auto qSplitSize = q_split_size > max_seqlen_q ? max_seqlen_q : q_split_size;
  auto kvSplitSize = block_size > max_seqlens_k ? max_seqlens_k : block_size;
  // kvSplitSize should be the same as block_size
  auto qSliceMax = (max_seqlen_q + qSplitSize - 1) / qSplitSize;
  auto kvSliceMax = (max_seqlens_k + kvSplitSize - 1) / kvSplitSize;

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  // ToDo(liangan1): align the scale semantic with other repo
  accum_t scaling_factor =
      calculate_scale(query, softmax_scale).as_float_unchecked();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);
  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * head_size;

  int64_t num_thread = at::get_num_threads();
  at::Tensor buf = at::empty(
      {num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::empty(
      {num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0},
      query.options());

  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key_cache.data_ptr<cache_t>();
  auto value_ptr = value_cache.data_ptr<cache_t>();
  auto cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int>();
  auto cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int>();
  auto block_table_ptr = block_table.data_ptr<int>();
  auto buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data =
      is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;
  auto alibi_slopes_ptr = alibi_slopes.has_value()
      ? alibi_slopes.value().data_ptr<float>()
      : nullptr;

#pragma omp parallel for collapse(3) schedule(static, 1)
  for (auto i = 0; i < batch_size; i++) {
    for (auto j = 0; j < num_heads; j++) {
      for (auto k = 0; k < qSliceMax; k++) {
        auto ompIdx = omp_get_thread_num();
        auto kv_head_id = j / kv_head_group_size;

        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;
        scalar_t* qk_reduced_data = is_reduced_type
            ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize
            : nullptr;

        // get the current query len
        int64_t qSize = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
        int64_t kvSize = cu_seqlens_k_ptr[i + 1] - cu_seqlens_k_ptr[i];

        int64_t context_len = kvSize - qSize; // computed context lens
        int64_t m = k * qSplitSize;
        if (m >= qSize) {
          continue;
        }
        int64_t qBlockSize = std::min(qSplitSize, qSize - m);

        // Initialize max and sum
        torch_ipex::cpu::kernel::fill_stub(
            qk_max_data, -std::numeric_limits<accum_t>::infinity(), qBlockSize);
        torch_ipex::cpu::kernel::fill_stub(
            qk_sum_data, static_cast<accum_t>(0), qBlockSize);
        torch_ipex::cpu::kernel::fill_stub(
            dst_data, static_cast<accum_t>(0), qBlockSize * head_size);
        int64_t num_keys =
            is_causal ? std::min(m + qBlockSize + context_len, kvSize) : kvSize;

        for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
          // get the physical block id of the key and value
          int64_t physical_block_id =
              block_table_ptr[i * max_num_blocks_per_seq + n / kvSplitSize];
          auto key_page_data = key_ptr + physical_block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          auto value_page_data = value_ptr +
              physical_block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
          // Calculate the scale * query * key
          // query block[qBlockSize, head_size], key block: [kvBlockSize,
          // head_size]
          _mkl_gemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasTrans,
              qBlockSize,
              kvBlockSize,
              head_size,
              static_cast<accum_t>(1),
              query_ptr + (cu_seqlens_q_ptr[i] + m) * q_strideN + j * q_strideH,
              q_strideN,
              key_page_data,
              head_size,
              static_cast<accum_t>(0),
              qk_data,
              kvSplitSize);

          // apply causal mask, fill unmasked position with -inf
          if (is_causal) {
            for (int64_t q = 0; q < qBlockSize; q++) {
              for (int64_t p = 0; p < kvBlockSize; p++) {
                if (m + q + context_len < n + p) {
                  qk_data[q * kvSplitSize + p] =
                      -std::numeric_limits<accum_t>::infinity();
                }
              }
            }
          }
          // Calculate max and sum of exp(val-max)
          for (int64_t q = 0; q < qBlockSize; q++) {
            accum_t tmp_max = -std::numeric_limits<accum_t>::infinity(),
                    tmp_sum = 0, exp_tmp = 0;

            _mul_reduce_max_fusion_kernel<accum_t>(
                qk_data + q * kvSplitSize,
                scaling_factor,
                kvBlockSize,
                qk_data + q * kvSplitSize,
                tmp_max);

            tmp_max = qk_max_data[q] > tmp_max ? qk_max_data[q] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel<accum_t, scalar_t>(
                qk_data + q * kvSplitSize,
                kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) +
                    q * kvSplitSize,
                tmp_sum);
            // exp_tmp <- exp(max[row] - max)
            exp_tmp = std::exp(qk_max_data[q] - tmp_max);
            // sum[row] <- sum + exp_tmp * sum[row]
            qk_sum_data[q] = tmp_sum + exp_tmp * qk_sum_data[q];
            // max[row] <- max
            qk_max_data[q] = tmp_max;
            // dst <- dst * exp_tmp
            if (n > 0) {
              at::vec::map<accum_t>(
                  [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                  dst_data + q * head_size,
                  dst_data + q * head_size,
                  head_size);
            }
          }

          // Calculate the sum of attn_weight * value

          _mkl_gemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              qBlockSize,
              head_size,
              kvBlockSize,
              static_cast<accum_t>(1),
              conditional_data_ptr(qk_data, qk_reduced_data),
              kvSplitSize,
              value_page_data,
              head_size,
              n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
              dst_data,
              head_size);
        }

        // copy the result to the output
        // dst<-dst/sum
        for (int64_t q = 0; q < qBlockSize; q++) {
          accum_t sum_reciprocal = 1 / qk_sum_data[q];
          at::vec::map<scalar_t>(
              [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
              out_ptr + (cu_seqlens_q_ptr[i] + m + q) * out_strideN +
                  j * out_strideH,
              dst_data + q * head_size,
              head_size);
        }
      }
    }
  }
}
void single_query_cached_kv_attention_kernel_impl(
    at::Tensor& out, // [num_seqs, num_heads, head_size]
    at::Tensor& query, // [num_seqs, num_heads, head_size]
    at::Tensor& key_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& value_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& head_mapping, // [num_heads]
    const double scale,
    at::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    at::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes,
    const double k_scale,
    const double v_scale) {
  RECORD_FUNCTION(
      "ipex::single_query_cached_kv_attention_kernel_impl",
      c10::ArrayRef<c10::IValue>({}));
  // dispatch kernel according to the data type of input tensor
  if (key_cache.scalar_type() == at::ScalarType::Float8_e5m2 &&
      out.scalar_type() == at::ScalarType::BFloat16) {
    single_query_cached_kv_attention_kernel<at::BFloat16, at::Float8_e5m2>(
        out,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        k_scale,
        v_scale);
  } else if (out.scalar_type() == at::ScalarType::Float) {
    single_query_cached_kv_attention_kernel<float, float>(
        out,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        k_scale,
        v_scale);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    single_query_cached_kv_attention_kernel<at::BFloat16, at::BFloat16>(
        out,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        k_scale,
        v_scale);
  } else if (out.scalar_type() == at::ScalarType::Half) {
    single_query_cached_kv_attention_kernel<at::Half, at::Half>(
        out,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        k_scale,
        v_scale);
  } else {
    TORCH_CHECK(
        false, "Unsupported data type for single_query_cached_kv_attention");
  }
}

// void reshape_and_cache_kernel
void reshape_and_cache_cpu_kernel_impl(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping,
    const double k_scale,
    const double v_scale) {
  TORCH_CHECK(
      key.scalar_type() == value.scalar_type(),
      "key and value should have the same data type");
  TORCH_CHECK(
      key_cache.scalar_type() == value_cache.scalar_type(),
      "key_cache and value_cache should have the same data type");
  TORCH_CHECK(
      slot_mapping.is_contiguous(), "slot_mapping should be contiguous");
  RECORD_FUNCTION(
      "ipex::reshape_and_cache_cpu_kernel_impl",
      c10::ArrayRef<c10::IValue>({}));
  if (key_cache.scalar_type() == at::ScalarType::Float8_e5m2 &&
      key.scalar_type() == at::ScalarType::Float) {
    reshape_and_cache_kernel<at::Float8_e5m2, float>(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale);
  } else if (
      key_cache.scalar_type() == at::ScalarType::Float8_e5m2 &&
      key.scalar_type() == at::ScalarType::BFloat16) {
    reshape_and_cache_kernel<at::Float8_e5m2, at::BFloat16>(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale);
  } else if (key.scalar_type() == at::ScalarType::Float) {
    reshape_and_cache_kernel<float, float>(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    reshape_and_cache_kernel<at::BFloat16, at::BFloat16>(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale);
  } else if (key.scalar_type() == at::ScalarType::Half) {
    reshape_and_cache_kernel<at::Half, at::Half>(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale);
  } else {
    TORCH_CHECK(false, "Unsupported data type for ipex::reshape_and_cache");
  }
}

void flash_attn_varlen_cpu_kernel_impl(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& cu_seqlens_q,
    at::Tensor& cu_seqlens_kv,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    const double softmax_scale,
    bool is_causal,
    at::Tensor& block_table,
    const c10::optional<at::Tensor>& alibi_slopes,
    const double k_scale,
    const double v_scale) {
  TORCH_CHECK(
      key.scalar_type() == value.scalar_type(),
      "key and value should have the same data type");
  TORCH_CHECK(
      !alibi_slopes.has_value(),
      "alibi_slopes is not supported for flash_attn_varlen yet");
  TORCH_CHECK(
      is_causal,
      "flash_attn_varlen_cpu_kernel_impl only supports causal attention, pls use the is_causal=True");
  TORCH_CHECK(
      query.scalar_type() == out.scalar_type(),
      "query and out should have the same data type");
  RECORD_FUNCTION(
      "ipex::flash_attn_varlen_cpu_kernel_impl",
      c10::ArrayRef<c10::IValue>({}));
  if (query.scalar_type() == at::ScalarType::Float) {
    if (max_seqlen_q >= 768) {
      flash_attn_varlen_kernel<float, float, 128>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    } else if (max_seqlen_q >= 192) {
      flash_attn_varlen_kernel<float, float, 64>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    } else {
      flash_attn_varlen_kernel<float, float, 32>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    }

  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    if (max_seqlen_q >= 768) {
      flash_attn_varlen_kernel<at::BFloat16, at::BFloat16, 128>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    } else if (max_seqlen_q >= 192) {
      flash_attn_varlen_kernel<at::BFloat16, at::BFloat16, 64>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    } else {
      flash_attn_varlen_kernel<at::BFloat16, at::BFloat16, 32>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    }

  } else if (query.scalar_type() == at::ScalarType::Half) {
    if (max_seqlen_q >= 768) {
      flash_attn_varlen_kernel<at::Half, at::Half, 128>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    } else if (max_seqlen_q >= 192) {
      flash_attn_varlen_kernel<at::Half, at::Half, 64>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    } else {
      flash_attn_varlen_kernel<at::Half, at::Half, 32>(
          out,
          query,
          key,
          value,
          cu_seqlens_q,
          cu_seqlens_kv,
          max_seqlen_q,
          max_seqlen_kv,
          softmax_scale,
          is_causal,
          block_table,
          alibi_slopes,
          k_scale,
          v_scale);
    }

  } else {
    TORCH_CHECK(false, "Unsupported data type for ipex::flash_attn_varlen");
  }
}

} // namespace

IPEX_REGISTER_DISPATCH(
    single_query_cached_kv_attention_kernel_stub,
    &single_query_cached_kv_attention_kernel_impl);
IPEX_REGISTER_DISPATCH(
    reshape_and_cache_kernel_stub,
    &reshape_and_cache_cpu_kernel_impl);
IPEX_REGISTER_DISPATCH(
    flash_attn_var_len_kernel_stub,
    &flash_attn_varlen_cpu_kernel_impl);

} // namespace cpu
} // namespace torch_ipex