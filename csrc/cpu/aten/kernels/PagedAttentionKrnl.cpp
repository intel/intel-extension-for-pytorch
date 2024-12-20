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
#include "csrc/cpu/tpp/woq/tla.h"
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
using namespace tpp;
namespace cpu {

namespace {

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
    const KT* k_cache_start,
    float* attn_w_pos,
    int64_t head_size) {
  attn_w_pos[0] = 0;
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::_reduce_head<QT, KT, KT>(
      q_ptr_start, k_cache_start, attn_w_pos, head_size, false, nullptr);
#else
  for (auto hsi = 0; hsi < head_size; hsi++) {
    attn_w_pos[0] += (float)q_ptr_start[hsi] * (float)k_cache_start[hsi];
  }
#endif
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
  auto vec_size = 16; // 512/32
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

template <typename OT, typename CT>
inline void mul_attenion_weights_and_value_of_head(
    const float& attn_w,
    const CT* v_cache_start,
    OT* attn_out_start,
    int64_t head_size,
    bool accumulated) {
  auto vec_size = 16; // 512/32
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::_mul_and_accumulate<CT, OT, CT>(
      attn_w,
      v_cache_start,
      attn_out_start,
      head_size,
      false,
      nullptr,
      accumulated);
#else
  for (hsi = 0; hsi < head_size; hsi++) {
    if (accumulated) {
      attn_out_start[hsi] += attn_w * (float)v_cache_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * (float)v_cache_start[hsi];
    }
  }
#endif
}

template <typename OT, typename CT>
inline typename std::enable_if_t<is_reduced_floating_point_v<CT>, void>
mul_attenion_weights_and_value_of_head(
    const CT& attn_w,
    const CT* v_cache_start,
    OT* attn_out_start,
    int64_t head_size,
    bool accumulated) {
  auto vec_size = 16; // 512/32
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::_mul_and_accumulate<CT, OT, CT>(
      attn_w,
      v_cache_start,
      attn_out_start,
      head_size,
      false,
      nullptr,
      accumulated);
#else
  for (hsi = 0; hsi < head_size; hsi++) {
    if (accumulated) {
      attn_out_start[hsi] += (float)attn_w * (float)v_cache_start[hsi];
    } else {
      attn_out_start[hsi] = (float)attn_w * (float)v_cache_start[hsi];
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
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @return prompt_block_nums  Prompt block number for each batch [batch_size].
 */
at::Tensor deduce_prompt(
    at::Tensor& block_tables) {
  // deduce beam size
  int beam_size = 1;
  int num_seqs = block_tables.size(0);
  int max_num_blocks_per_seq = block_tables.size(1);
  auto block_tables_ptr = block_tables.data_ptr<int>();
  auto first_block_id = block_tables_ptr[0];
  for (int i=1; i<block_tables.size(0); i++) {
    if (block_tables_ptr[i * max_num_blocks_per_seq] != first_block_id) {
      break;
    }
    beam_size++;
  }
  TORCH_CHECK(num_seqs % beam_size == 0, "num seqs is not divisible by beam size.");
  int batch_size = num_seqs / beam_size;
  // deduce prompt len
  auto prompt_block_nums = at::empty(
      {batch_size},
      block_tables.options().dtype());
  auto prompt_block_nums_ptr = prompt_block_nums.data_ptr<int>();
#pragma omp parallel for
  for (int i=0; i<batch_size; i++) {
    // find prompt blocks
    int final_prompt_blocks = max_num_blocks_per_seq;
    auto block_tables_offset = block_tables_ptr + i * beam_size * max_num_blocks_per_seq;
    for (int j=1; j<beam_size; j++) {
      int curr_blocks = 0;
      while (curr_blocks < max_num_blocks_per_seq
          && (block_tables_offset[curr_blocks] == block_tables_offset[j * max_num_blocks_per_seq + curr_blocks])) {
        curr_blocks++;
      }
      final_prompt_blocks = std::min(final_prompt_blocks, curr_blocks);
    }
    prompt_block_nums_ptr[i] = final_prompt_blocks;
  }
  return prompt_block_nums;
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
 * shape should be [num_blocks, num_key_value_heads, block_size, head_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, num_key_value_heads, block_size, head_size].
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [num_seqs].
 * @param block_size    The block size which means the number of token in every
 * block.
 * @param max_context_len Maximum context length.
 * @param alibi_slopes  Optional tensor of alibi slopes with the shape of
 * (num_heads).
 */
template <typename scalar_t>
void single_query_cached_kv_attention_vnni_v_kernel(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const double scale,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes) {
  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
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

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  auto prompt_block_nums = deduce_prompt(block_tables);
  auto prompt_block_nums_ptr = prompt_block_nums.data_ptr<int>();
  int batch_size = prompt_block_nums.size(0);
  int beam_size = num_seqs / batch_size;

  if (alibi_slopes.has_value()) {
    auto alibi_slopes_size = alibi_slopes.value().size(0);
    TORCH_CHECK(
        alibi_slopes_size == num_heads,
        "alibi_slopes size is not equal to num_heads");
  }

  auto thread_numbers = omp_get_max_threads();
  auto attn_weights = at::empty(
      {num_seqs, num_heads, max_context_len},
      query.options().dtype(at::ScalarType::Float));
  auto attn_weights_reduced = at::empty(
      {num_seqs, num_heads, is_reduced_type ? max_context_len : 0},
      query.options());
  auto fp32_attn_outs = at::zeros({num_seqs, num_heads, head_size}, at::kFloat);
  at::Tensor key_t_reorder = at::empty(
        {thread_numbers, head_size, block_size},
        c10::CppTypeToScalarType<scalar_t>::value);
  at::Tensor attn_pad = at::empty(
        {thread_numbers, block_size},
        at::kFloat);
  at::Tensor attn_pad_reduced = at::empty(
        {thread_numbers, is_reduced_type ? block_size : 0},
        query.options());

  // strides
  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = key_cache.stride(2);
  auto kv_block_strideH = key_cache.stride(1);
  auto fp32_attn_out_strideN = fp32_attn_outs.stride(0);
  auto fp32_attn_out_strideH = fp32_attn_outs.stride(1);
  auto attn_out_strideN = out.stride(0);
  auto attn_out_strideH = out.stride(1);
  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);
  auto attn_weights_strideN = attn_weights.stride(0);
  auto attn_weights_strideH = attn_weights.stride(1);
  auto key_t_reorder_strideT = key_t_reorder.stride(0);

  // get pointers
  auto attn_weights_ptr = attn_weights.data_ptr<float>();
  auto attn_weights_reduced_ptr = is_reduced_type ?
        attn_weights_reduced.data_ptr<scalar_t>() : nullptr;
  auto fp32_attn_out_ptr = fp32_attn_outs.data_ptr<float>();
  auto key_t_reorder_ptr = key_t_reorder.data_ptr<scalar_t>();
  auto attn_pad_ptr = attn_pad.data_ptr<float>();
  auto attn_pad_reduced_ptr = is_reduced_type ?
        attn_pad_reduced.data_ptr<scalar_t>() : nullptr;

  // TODO: support padding
  TORCH_CHECK(
    head_size % 2 == 0,
    "head size is not even");
  auto k_xform = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ block_size,
          /*in_cols*/ head_size,
          /*out_rows*/ head_size,
          /*out_cols*/ block_size,
          /*ldi*/ kv_block_strideP,
          /*ldo*/ block_size,
          /*xtype*/ XformTPP::XFORM_XPOSE_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  auto qk_gemm_prompt = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ beam_size,
      /*N*/ block_size,
      /*K*/ head_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ q_strideN,
      /*ldb*/ block_size,
      /*ldc*/ attn_weights_strideN,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto qk_gemm_rest = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ 1,
      /*N*/ block_size,
      /*K*/ head_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ q_strideN,
      /*ldb*/ block_size,
      /*ldc*/ attn_weights_strideN,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto av_gemm_prompt = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ beam_size,
      /*N*/ head_size,
      /*K*/ block_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ attn_weights_strideN,
      /*ldb*/ head_size,
      /*ldc*/ fp32_attn_out_strideN,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto av_gemm_rest = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ 1,
      /*N*/ head_size,
      /*K*/ block_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ attn_weights_strideN,
      /*ldb*/ head_size,
      /*ldc*/ fp32_attn_out_strideN,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto head_id = 0; head_id < num_heads; head_id++) {
    for (auto batch_id = 0; batch_id < batch_size; batch_id++) {
      auto ompIdx = omp_get_thread_num();
      auto k_cache_reorder = key_t_reorder_ptr + ompIdx * key_t_reorder_strideT;
      int kv_head_id = head_id / kv_head_group_size;
      auto seq_id_start = batch_id * beam_size;
      int context_len = context_lens_ptr[seq_id_start]; // the beams in one batch have the same context_len
      int complete_block_num = int(context_len / block_size);
      int prompt_block_num = std::min(prompt_block_nums_ptr[batch_id], complete_block_num);
      int complete_token_length = complete_block_num * block_size;
      auto query_ptr_start = query_ptr + seq_id_start * q_strideN + head_id * q_strideH;
      auto attn_weights_ptr_start = attn_weights_ptr
          + seq_id_start * attn_weights_strideN
          + head_id * attn_weights_strideH;
      // for prompt
      for (auto seq_block_id = 0; seq_block_id < prompt_block_num; seq_block_id++) {
        auto block_id = block_tables_ptr
            [seq_id_start * max_num_blocks_per_seq + seq_block_id];
        auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
            kv_head_id * kv_block_strideH;
        k_xform(k_cache_start, k_cache_reorder);
        qk_gemm_prompt(
            query_ptr_start,
            k_cache_reorder,
            attn_weights_ptr_start + seq_block_id * block_size,
            1);
      }
      // for the rest complete blocks
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto q_ptr_start = query_ptr_start + beam_id * q_strideN;
        for (auto seq_block_id = prompt_block_num; seq_block_id < complete_block_num; seq_block_id++) {
          auto attn_w_pos = attn_weights_ptr_start
              + beam_id * attn_weights_strideN
              + seq_block_id * block_size;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + seq_block_id];
          auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          k_xform(k_cache_start, k_cache_reorder);
          qk_gemm_rest(
              q_ptr_start,
              k_cache_reorder,
              attn_w_pos,
              1);
        }
      }
      // for the rest tail
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto q_ptr_start = query_ptr_start + beam_id * q_strideN;
        for (auto token_id = complete_token_length; token_id < context_len; token_id++) {
          auto attn_w_pos = attn_weights_ptr_start
              + beam_id * attn_weights_strideN
              + token_id;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + token_id / block_size];
          auto block_offset = token_id % block_size;
          auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
              block_offset * kv_block_strideP +
              kv_head_id * kv_block_strideH;
          reduce_head<scalar_t, scalar_t>(
              q_ptr_start, k_cache_start, attn_w_pos, head_size);
        }
      }
    }
  }

// div+add+softmax
#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto max_val = -10000.0f;
      float sum = 0.0f;
      auto context_len = context_lens_ptr[seq_id];
      auto attn_w_start = attn_weights_ptr
          + seq_id * attn_weights_strideN
          + head_id * attn_weights_strideH;
      auto attn_w_start_reduced = is_reduced_type ? attn_weights_reduced_ptr
          + seq_id * attn_weights_strideN
          + head_id * attn_weights_strideH : nullptr;
#if defined(CPU_CAPABILITY_AVX512)
      if (alibi_slopes_ptr != nullptr) {
        auto alibi_slope = alibi_slopes_ptr[head_id];
        torch_ipex::cpu::kernel::
            _dil_div_add_alibi_and_reduce_max_fusion_kernel<float>(
                attn_w_start,
                scale,
                context_len,
                attn_w_start,
                max_val,
                alibi_slope,
                true);
      } else {
        torch_ipex::cpu::kernel::
            _dil_div_add_alibi_and_reduce_max_fusion_kernel<float>(
                attn_w_start,
                scale,
                context_len,
                attn_w_start,
                max_val,
                1,
                false);
      }
      torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(
          attn_w_start, context_len, attn_w_start, max_val);
      torch_ipex::cpu::kernel::_dil_normalization_kernel<scalar_t>(
          attn_w_start, max_val, context_len,
          conditional_data_ptr(attn_w_start, attn_w_start_reduced));

#else
      // div+add+softmax
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = attn_w_start[token_id] * scale;
        if (alibi_slopes_ptr != nullptr) {
          auto alibi_slope = alibi_slopes_ptr[head_id];
          auto alibi_slopes_val =
              alibi_slope * (token_id + 1 - context_lens_ptr[seq_id]);
          attn_w_start[token_id] = attn_w_start[token_id] + alibi_slopes_val;
        }
        if (attn_w_start[token_id] > max_val) {
          max_val = attn_w_start[token_id];
        }
      }
      // exp and sum
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = exp(attn_w_start[token_id] - max_val);
        sum += attn_w_start[token_id];
      }
      // normalize
      if (is_reduced_type) {
        for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
          attn_w_start_reduced[token_id] = (scalar_t)(attn_w_start[token_id] / sum);
        }
      } else {
        for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
          attn_w_start[token_id] = attn_w_start[token_id] / sum;
        }
      }
#endif
    }
  }

// mul and accumulate
#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto head_id = 0; head_id < num_heads; head_id++) {
    for (auto batch_id = 0; batch_id < batch_size; batch_id++) {
      auto ompIdx = omp_get_thread_num();
      auto attn_pad_start = conditional_data_ptr(attn_pad_ptr, attn_pad_reduced_ptr)
            + ompIdx * block_size;
      int kv_head_id = head_id / kv_head_group_size;
      int context_len = context_lens_ptr[batch_id * beam_size]; // the beams in one batch have the same context_len
      int complete_block_num = int(context_len / block_size);
      int prompt_block_num = std::min(prompt_block_nums_ptr[batch_id], complete_block_num);
      if (context_len % block_size != 0) {
        complete_block_num ++;
      }
      int complete_token_length = complete_block_num * block_size;
      auto seq_id_start = batch_id * beam_size;
      auto attn_weights_start = conditional_data_ptr(attn_weights_ptr, attn_weights_reduced_ptr)
            + seq_id_start * attn_weights_strideN +
               head_id * attn_weights_strideH;
      auto fp32_attn_out_start = fp32_attn_out_ptr + seq_id_start * fp32_attn_out_strideN
              + head_id * fp32_attn_out_strideH;
      // for prompt
      for (auto seq_block_id = 0; seq_block_id < prompt_block_num; seq_block_id++) {
        auto block_id = block_tables_ptr
            [seq_id_start * max_num_blocks_per_seq + seq_block_id];
        auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
            kv_head_id * kv_block_strideH;
        av_gemm_prompt(
            attn_weights_start + seq_block_id * block_size,
            v_cache_start,
            fp32_attn_out_start,
            1);
      }
      // for the rest complete blocks
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto attn_out_start = fp32_attn_out_start +
            beam_id * fp32_attn_out_strideN;
        for (auto seq_block_id = prompt_block_num; seq_block_id < complete_block_num; seq_block_id++) {
          auto attn_w = attn_weights_start
              + beam_id * attn_weights_strideN + seq_block_id * block_size;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + seq_block_id];
          auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          if (seq_block_id == complete_block_num - 1 && (context_len % block_size) != 0) {
            auto cnt = context_len % block_size;
            torch_ipex::cpu::kernel::move_ker<scalar_t, scalar_t>(
              attn_pad_start, attn_w, cnt);
            torch_ipex::cpu::kernel::zero_ker<scalar_t>(
              attn_pad_start + cnt, block_size - cnt);
            av_gemm_rest(
              attn_pad_start,
              v_cache_start,
              attn_out_start,
              1);
          } else {
            av_gemm_rest(
              attn_w,
              v_cache_start,
              attn_out_start,
              1);
          }
        }
        // write to the output after the last token is done
        auto out_start = out_ptr +
          seq_id * attn_out_strideN + head_id * attn_out_strideH;
        torch_ipex::cpu::kernel::move_ker<scalar_t, float>(
            out_start, attn_out_start, head_size);
      }
    }
  }

} // single_query_cached_kv_attention_vnni_v_kernel

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
 * shape should be [num_blocks, num_key_value_heads, head_size, block_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, num_key_value_heads, block_size, head_size].
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [num_seqs].
 * @param block_size    The block size which means the number of token in every
 * block.
 * @param max_context_len Maximum context length.
 * @param alibi_slopes  Optional tensor of alibi slopes with the shape of
 * (num_heads).
 */
template <typename scalar_t>
void single_query_cached_kv_attention_vnni_kernel(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const double scale,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes) {
  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
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

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  auto prompt_block_nums = deduce_prompt(block_tables);
  auto prompt_block_nums_ptr = prompt_block_nums.data_ptr<int>();
  int batch_size = prompt_block_nums.size(0);
  int beam_size = num_seqs / batch_size;

  if (alibi_slopes.has_value()) {
    auto alibi_slopes_size = alibi_slopes.value().size(0);
    TORCH_CHECK(
        alibi_slopes_size == num_heads,
        "alibi_slopes size is not equal to num_heads");
  }

  auto thread_numbers = omp_get_max_threads();
  auto attn_weights = at::empty(
      {num_seqs, num_heads, max_context_len},
      query.options().dtype(at::ScalarType::Float));
  auto attn_weights_reduced = at::empty(
      {num_seqs, num_heads, is_reduced_type ? max_context_len : 0},
      query.options());
  auto fp32_attn_outs = at::zeros({num_seqs, num_heads, head_size}, at::kFloat);
  at::Tensor attn_pad = at::empty(
        {thread_numbers, block_size},
        at::kFloat);
  at::Tensor attn_pad_reduced = at::empty(
        {thread_numbers, is_reduced_type ? block_size : 0},
        query.options());

  // strides
  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = value_cache.stride(2);
  auto kv_block_strideS = key_cache.stride(2);
  auto kv_block_strideH = key_cache.stride(1);
  auto fp32_attn_out_strideN = fp32_attn_outs.stride(0);
  auto fp32_attn_out_strideH = fp32_attn_outs.stride(1);
  auto attn_out_strideN = out.stride(0);
  auto attn_out_strideH = out.stride(1);
  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);
  auto attn_weights_strideN = attn_weights.stride(0);
  auto attn_weights_strideH = attn_weights.stride(1);

  // get pointers
  auto attn_weights_ptr = attn_weights.data_ptr<float>();
  auto attn_weights_reduced_ptr = is_reduced_type ?
        attn_weights_reduced.data_ptr<scalar_t>() : nullptr;
  auto fp32_attn_out_ptr = fp32_attn_outs.data_ptr<float>();
  auto attn_pad_ptr = attn_pad.data_ptr<float>();
  auto attn_pad_reduced_ptr = is_reduced_type ?
        attn_pad_reduced.data_ptr<scalar_t>() : nullptr;

  auto qk_gemm_prompt = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ beam_size,
      /*N*/ block_size,
      /*K*/ head_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ q_strideN,
      /*ldb*/ block_size,
      /*ldc*/ attn_weights_strideN,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto qk_gemm_rest = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ 1,
      /*N*/ block_size,
      /*K*/ head_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ q_strideN,
      /*ldb*/ block_size,
      /*ldc*/ attn_weights_strideN,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto av_gemm_prompt = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ beam_size,
      /*N*/ head_size,
      /*K*/ block_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ attn_weights_strideN,
      /*ldb*/ head_size,
      /*ldc*/ fp32_attn_out_strideN,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto av_gemm_rest = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ 1,
      /*N*/ head_size,
      /*K*/ block_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ attn_weights_strideN,
      /*ldb*/ head_size,
      /*ldc*/ fp32_attn_out_strideN,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto head_id = 0; head_id < num_heads; head_id++) {
    for (auto batch_id = 0; batch_id < batch_size; batch_id++) {
      auto ompIdx = omp_get_thread_num();
      auto attn_pad_start = attn_pad_ptr + ompIdx * block_size;
      int kv_head_id = head_id / kv_head_group_size;
      auto seq_id_start = batch_id * beam_size;
      int context_len = context_lens_ptr[seq_id_start]; // the beams in one batch have the same context_len
      int complete_block_num = int(context_len / block_size);
      int prompt_block_num = std::min(prompt_block_nums_ptr[batch_id], complete_block_num);
      if (context_len % block_size != 0) {
        complete_block_num ++;
      }
      auto query_ptr_start = query_ptr + seq_id_start * q_strideN + head_id * q_strideH;
      auto attn_weights_ptr_start = attn_weights_ptr
          + seq_id_start * attn_weights_strideN
          + head_id * attn_weights_strideH;
      // for prompt
      for (auto seq_block_id = 0; seq_block_id < prompt_block_num; seq_block_id++) {
        auto block_id = block_tables_ptr
            [seq_id_start * max_num_blocks_per_seq + seq_block_id];
        auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
            kv_head_id * kv_block_strideH;
        qk_gemm_prompt(
            query_ptr_start,
            k_cache_start,
            attn_weights_ptr_start + seq_block_id * block_size,
            1);
      }
      // for the rest complete blocks
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto q_ptr_start = query_ptr_start + beam_id * q_strideN;
        for (auto seq_block_id = prompt_block_num; seq_block_id < complete_block_num; seq_block_id++) {
          auto attn_w_pos = attn_weights_ptr_start
              + beam_id * attn_weights_strideN
              + seq_block_id * block_size;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + seq_block_id];
          auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          if (seq_block_id == complete_block_num - 1 && (context_len % block_size) != 0) {
            auto cnt = context_len % block_size;
            torch_ipex::cpu::kernel::zero_ker<float>(
              attn_pad_start, block_size);
            qk_gemm_rest(
              q_ptr_start,
              k_cache_start,
              attn_pad_start,
              1);
            torch_ipex::cpu::kernel::move_ker<float, float>(
              attn_w_pos, attn_pad_start, cnt);
          } else {
            qk_gemm_rest(
              q_ptr_start,
              k_cache_start,
              attn_w_pos,
              1);
          }
        }
      }
    }
  }

// div+add+softmax
#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto max_val = -10000.0f;
      float sum = 0.0f;
      auto context_len = context_lens_ptr[seq_id];
      auto attn_w_start = attn_weights_ptr
          + seq_id * attn_weights_strideN
          + head_id * attn_weights_strideH;
      auto attn_w_start_reduced = is_reduced_type ? attn_weights_reduced_ptr
          + seq_id * attn_weights_strideN
          + head_id * attn_weights_strideH : nullptr;
#if defined(CPU_CAPABILITY_AVX512)
      if (alibi_slopes_ptr != nullptr) {
        auto alibi_slope = alibi_slopes_ptr[head_id];
        torch_ipex::cpu::kernel::
            _dil_div_add_alibi_and_reduce_max_fusion_kernel<float>(
                attn_w_start,
                scale,
                context_len,
                attn_w_start,
                max_val,
                alibi_slope,
                true);
      } else {
        torch_ipex::cpu::kernel::
            _dil_div_add_alibi_and_reduce_max_fusion_kernel<float>(
                attn_w_start,
                scale,
                context_len,
                attn_w_start,
                max_val,
                1,
                false);
      }
      torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(
          attn_w_start, context_len, attn_w_start, max_val);
      torch_ipex::cpu::kernel::_dil_normalization_kernel<scalar_t>(
          attn_w_start, max_val, context_len,
          conditional_data_ptr(attn_w_start, attn_w_start_reduced));

#else
      // div+add+softmax
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = attn_w_start[token_id] * scale;
        if (alibi_slopes_ptr != nullptr) {
          auto alibi_slope = alibi_slopes_ptr[head_id];
          auto alibi_slopes_val =
              alibi_slope * (token_id + 1 - context_lens_ptr[seq_id]);
          attn_w_start[token_id] = attn_w_start[token_id] + alibi_slopes_val;
        }
        if (attn_w_start[token_id] > max_val) {
          max_val = attn_w_start[token_id];
        }
      }
      // exp and sum
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = exp(attn_w_start[token_id] - max_val);
        sum += attn_w_start[token_id];
      }
      // normalize
      if (is_reduced_type) {
        for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
          attn_w_start_reduced[token_id] = (scalar_t)(attn_w_start[token_id] / sum);
        }
      } else {
        for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
          attn_w_start[token_id] = attn_w_start[token_id] / sum;
        }
      }
#endif
    }
  }

// mul and accumulate
#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto head_id = 0; head_id < num_heads; head_id++) {
    for (auto batch_id = 0; batch_id < batch_size; batch_id++) {
      auto ompIdx = omp_get_thread_num();
      auto attn_pad_start = conditional_data_ptr(attn_pad_ptr, attn_pad_reduced_ptr)
            + ompIdx * block_size;
      int kv_head_id = head_id / kv_head_group_size;
      int context_len = context_lens_ptr[batch_id * beam_size]; // the beams in one batch have the same context_len
      int complete_block_num = int(context_len / block_size);
      int prompt_block_num = std::min(prompt_block_nums_ptr[batch_id], complete_block_num);
      if (context_len % block_size != 0) {
        complete_block_num ++;
      }
      // int complete_token_length = complete_block_num * block_size;
      auto seq_id_start = batch_id * beam_size;
      auto attn_weights_start = conditional_data_ptr(attn_weights_ptr, attn_weights_reduced_ptr)
            + seq_id_start * attn_weights_strideN +
               head_id * attn_weights_strideH;
      auto fp32_attn_out_start = fp32_attn_out_ptr + seq_id_start * fp32_attn_out_strideN
              + head_id * fp32_attn_out_strideH;
      // for prompt
      for (auto seq_block_id = 0; seq_block_id < prompt_block_num; seq_block_id++) {
        auto block_id = block_tables_ptr
            [seq_id_start * max_num_blocks_per_seq + seq_block_id];
        auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
            kv_head_id * kv_block_strideH;
        av_gemm_prompt(
            attn_weights_start + seq_block_id * block_size,
            v_cache_start,
            fp32_attn_out_start,
            1);
      }
      // for the rest complete blocks
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto attn_out_start = fp32_attn_out_start +
            beam_id * fp32_attn_out_strideN;
        for (auto seq_block_id = prompt_block_num; seq_block_id < complete_block_num; seq_block_id++) {
          auto attn_w = attn_weights_start
              + beam_id * attn_weights_strideN + seq_block_id * block_size;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + seq_block_id];
          auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          if (seq_block_id == complete_block_num - 1 && (context_len % block_size) != 0) {
            auto cnt = context_len % block_size;
            torch_ipex::cpu::kernel::move_ker<scalar_t, scalar_t>(
              attn_pad_start, attn_w, cnt);
            torch_ipex::cpu::kernel::zero_ker<scalar_t>(
              attn_pad_start + cnt, block_size - cnt);
            av_gemm_rest(
              attn_pad_start,
              v_cache_start,
              attn_out_start,
              1);
          } else {
            av_gemm_rest(
              attn_w,
              v_cache_start,
              attn_out_start,
              1);
          }
        }
        // write to the output after the last token is done
        auto out_start = out_ptr +
          seq_id * attn_out_strideN + head_id * attn_out_strideH;
        torch_ipex::cpu::kernel::move_ker<scalar_t, float>(
            out_start, attn_out_start, head_size);
      }
    }
  }

} // single_query_cached_kv_attention_vnni_kernel

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
 * shape should be [num_blocks, num_key_value_heads, block_size, head_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, num_key_value_heads, block_size, head_size].
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [num_seqs].
 * @param block_size    The block size which means the number of token in every
 * block.
 * @param max_context_len Maximum context length.
 * @param alibi_slopes  Optional tensor of alibi slopes with the shape of
 * (num_heads).
 */
template <typename scalar_t>
void single_query_cached_kv_attention_opt_kernel(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const double scale,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes) {
  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
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

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  auto prompt_block_nums = deduce_prompt(block_tables);
  auto prompt_block_nums_ptr = prompt_block_nums.data_ptr<int>();
  int batch_size = prompt_block_nums.size(0);
  int beam_size = num_seqs / batch_size;

  if (alibi_slopes.has_value()) {
    auto alibi_slopes_size = alibi_slopes.value().size(0);
    TORCH_CHECK(
        alibi_slopes_size == num_heads,
        "alibi_slopes size is not equal to num_heads");
  }

  auto thread_numbers = omp_get_max_threads();
  auto attn_weights = at::empty(
      {num_seqs, num_heads, max_context_len},
      query.options().dtype(at::ScalarType::Float));
  auto attn_weights_reduced = at::empty(
      {num_seqs, num_heads, is_reduced_type ? max_context_len : 0},
      query.options());
  auto fp32_attn_outs = at::zeros({num_seqs, num_heads, head_size}, at::kFloat);
  at::Tensor key_t_reorder = at::empty(
        {thread_numbers, head_size, block_size},
        c10::CppTypeToScalarType<scalar_t>::value);
  at::Tensor value_t_reorder = at::empty(
        {thread_numbers, block_size, head_size},
        c10::CppTypeToScalarType<scalar_t>::value);

  // strides
  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = key_cache.stride(2);
  auto kv_block_strideH = key_cache.stride(1);
  auto fp32_attn_out_strideN = fp32_attn_outs.stride(0);
  auto fp32_attn_out_strideH = fp32_attn_outs.stride(1);
  auto attn_out_strideN = out.stride(0);
  auto attn_out_strideH = out.stride(1);
  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);
  auto attn_weights_strideN = attn_weights.stride(0);
  auto attn_weights_strideH = attn_weights.stride(1);
  auto key_t_reorder_strideT = key_t_reorder.stride(0);
  auto value_t_reorder_strideT = value_t_reorder.stride(0);

  // get pointers
  auto attn_weights_ptr = attn_weights.data_ptr<float>();
  auto attn_weights_reduced_ptr = is_reduced_type ?
        attn_weights_reduced.data_ptr<scalar_t>() : nullptr;
  auto fp32_attn_out_ptr = fp32_attn_outs.data_ptr<float>();
  auto key_t_reorder_ptr = key_t_reorder.data_ptr<scalar_t>();
  auto value_t_reorder_ptr = value_t_reorder.data_ptr<scalar_t>();

  // TODO: support padding
  TORCH_CHECK(
    head_size % 2 == 0,
    "head size is not even");
  auto k_xform = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ block_size,
          /*in_cols*/ head_size,
          /*out_rows*/ head_size,
          /*out_cols*/ block_size,
          /*ldi*/ kv_block_strideP,
          /*ldo*/ block_size,
          /*xtype*/ XformTPP::XFORM_XPOSE_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  // TODO: support padding
  TORCH_CHECK(
    block_size % 2 == 0,
    "block size is not even");
  auto v_xform = SCOPEIT(
      XformExtTPP<scalar_t>(
          /*in_rows*/ block_size,
          /*in_cols*/ head_size,
          /*out_rows*/ block_size,
          /*out_cols*/ head_size,
          /*ldi*/ kv_block_strideP,
          /*ldo*/ head_size,
          /*xtype*/ XformTPP::XFORM_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  auto qk_gemm_prompt = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ beam_size,
      /*N*/ block_size,
      /*K*/ head_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ q_strideN,
      /*ldb*/ block_size,
      /*ldc*/ attn_weights_strideN,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto qk_gemm_rest = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ 1,
      /*N*/ block_size,
      /*K*/ head_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ q_strideN,
      /*ldb*/ block_size,
      /*ldc*/ attn_weights_strideN,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto av_gemm_prompt = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ beam_size,
      /*N*/ head_size,
      /*K*/ block_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ attn_weights_strideN,
      /*ldb*/ head_size,
      /*ldc*/ fp32_attn_out_strideN,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

  auto av_gemm_rest = SCOPEITGEMM((BrgemmTPP<scalar_t, float>(
      /*M*/ 1,
      /*N*/ head_size,
      /*K*/ block_size,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ attn_weights_strideN,
      /*ldb*/ head_size,
      /*ldc*/ fp32_attn_out_strideN,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1,
      /*b_vnni*/ 1)));

#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto head_id = 0; head_id < num_heads; head_id++) {
    for (auto batch_id = 0; batch_id < batch_size; batch_id++) {
      auto ompIdx = omp_get_thread_num();
      auto k_cache_reorder = key_t_reorder_ptr + ompIdx * key_t_reorder_strideT;
      int kv_head_id = head_id / kv_head_group_size;
      auto seq_id_start = batch_id * beam_size;
      int context_len = context_lens_ptr[seq_id_start]; // the beams in one batch have the same context_len
      int complete_block_num = int(context_len / block_size);
      int prompt_block_num = std::min(prompt_block_nums_ptr[batch_id], complete_block_num);
      int complete_token_length = complete_block_num * block_size;
      auto query_ptr_start = query_ptr + seq_id_start * q_strideN + head_id * q_strideH;
      auto attn_weights_ptr_start = attn_weights_ptr
          + seq_id_start * attn_weights_strideN
          + head_id * attn_weights_strideH;
      // for prompt
      for (auto seq_block_id = 0; seq_block_id < prompt_block_num; seq_block_id++) {
        auto block_id = block_tables_ptr
            [seq_id_start * max_num_blocks_per_seq + seq_block_id];
        auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
            kv_head_id * kv_block_strideH;
        k_xform(k_cache_start, k_cache_reorder);
        qk_gemm_prompt(
            query_ptr_start,
            k_cache_reorder,
            attn_weights_ptr_start + seq_block_id * block_size,
            1);
      }
      // for the rest complete blocks
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto q_ptr_start = query_ptr_start + beam_id * q_strideN;
        for (auto seq_block_id = prompt_block_num; seq_block_id < complete_block_num; seq_block_id++) {
          auto attn_w_pos = attn_weights_ptr_start
              + beam_id * attn_weights_strideN
              + seq_block_id * block_size;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + seq_block_id];
          auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          k_xform(k_cache_start, k_cache_reorder);
          qk_gemm_rest(
              q_ptr_start,
              k_cache_reorder,
              attn_w_pos,
              1);
        }
      }
      // for the rest tail
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto q_ptr_start = query_ptr_start + beam_id * q_strideN;
        for (auto token_id = complete_token_length; token_id < context_len; token_id++) {
          auto attn_w_pos = attn_weights_ptr_start
              + beam_id * attn_weights_strideN
              + token_id;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + token_id / block_size];
          auto block_offset = token_id % block_size;
          auto k_cache_start = key_cache_ptr + block_id * kv_block_strideN +
              block_offset * kv_block_strideP +
              kv_head_id * kv_block_strideH;
          reduce_head<scalar_t, scalar_t>(
              q_ptr_start, k_cache_start, attn_w_pos, head_size);
        }
      }
    }
  }

// div+add+softmax
#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto max_val = -10000.0f;
      float sum = 0.0f;
      auto context_len = context_lens_ptr[seq_id];
      auto attn_w_start = attn_weights_ptr
          + seq_id * attn_weights_strideN
          + head_id * attn_weights_strideH;
      auto attn_w_start_reduced = is_reduced_type ? attn_weights_reduced_ptr
          + seq_id * attn_weights_strideN
          + head_id * attn_weights_strideH : nullptr;
#if defined(CPU_CAPABILITY_AVX512)
      if (alibi_slopes_ptr != nullptr) {
        auto alibi_slope = alibi_slopes_ptr[head_id];
        torch_ipex::cpu::kernel::
            _dil_div_add_alibi_and_reduce_max_fusion_kernel<float>(
                attn_w_start,
                scale,
                context_len,
                attn_w_start,
                max_val,
                alibi_slope,
                true);
      } else {
        torch_ipex::cpu::kernel::
            _dil_div_add_alibi_and_reduce_max_fusion_kernel<float>(
                attn_w_start,
                scale,
                context_len,
                attn_w_start,
                max_val,
                1,
                false);
      }
      torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(
          attn_w_start, context_len, attn_w_start, max_val);
      torch_ipex::cpu::kernel::_dil_normalization_kernel<scalar_t>(
          attn_w_start, max_val, context_len,
          conditional_data_ptr(attn_w_start, attn_w_start_reduced));

#else
      // div+add+softmax
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = attn_w_start[token_id] * scale;
        if (alibi_slopes_ptr != nullptr) {
          auto alibi_slope = alibi_slopes_ptr[head_id];
          auto alibi_slopes_val =
              alibi_slope * (token_id + 1 - context_lens_ptr[seq_id]);
          attn_w_start[token_id] = attn_w_start[token_id] + alibi_slopes_val;
        }
        if (attn_w_start[token_id] > max_val) {
          max_val = attn_w_start[token_id];
        }
      }
      // exp and sum
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = exp(attn_w_start[token_id] - max_val);
        sum += attn_w_start[token_id];
      }
      // normalize
      if (is_reduced_type) {
        for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
          attn_w_start_reduced[token_id] = (scalar_t)(attn_w_start[token_id] / sum);
        }
      } else {
        for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
          attn_w_start[token_id] = attn_w_start[token_id] / sum;
        }
      }
#endif
    }
  }

// mul and accumulate
#pragma omp parallel for collapse(2) schedule(static, 1)
  for (auto head_id = 0; head_id < num_heads; head_id++) {
    for (auto batch_id = 0; batch_id < batch_size; batch_id++) {
      auto ompIdx = omp_get_thread_num();
      auto v_cache_reorder = value_t_reorder_ptr + ompIdx * value_t_reorder_strideT;
      int kv_head_id = head_id / kv_head_group_size;
      int context_len = context_lens_ptr[batch_id * beam_size]; // the beams in one batch have the same context_len
      int complete_block_num = int(context_len / block_size);
      int prompt_block_num = std::min(prompt_block_nums_ptr[batch_id], complete_block_num);
      int complete_token_length = complete_block_num * block_size;
      auto seq_id_start = batch_id * beam_size;
      auto attn_weights_start = conditional_data_ptr(attn_weights_ptr, attn_weights_reduced_ptr)
            + seq_id_start * attn_weights_strideN +
               head_id * attn_weights_strideH;
      auto fp32_attn_out_start = fp32_attn_out_ptr + seq_id_start * fp32_attn_out_strideN
              + head_id * fp32_attn_out_strideH;
      // for prompt
      for (auto seq_block_id = 0; seq_block_id < prompt_block_num; seq_block_id++) {
        auto block_id = block_tables_ptr
            [seq_id_start * max_num_blocks_per_seq + seq_block_id];
        auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
            kv_head_id * kv_block_strideH;
        v_xform(v_cache_start, v_cache_reorder);
        av_gemm_prompt(
            attn_weights_start + seq_block_id * block_size,
            v_cache_reorder,
            fp32_attn_out_start,
            1);
      }
      // for the rest complete blocks
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto attn_out_start = fp32_attn_out_start +
            beam_id * fp32_attn_out_strideN;
        for (auto seq_block_id = prompt_block_num; seq_block_id < complete_block_num; seq_block_id++) {
          auto attn_w = attn_weights_start
              + beam_id * attn_weights_strideN + seq_block_id * block_size;
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + seq_block_id];
          auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
              kv_head_id * kv_block_strideH;
          v_xform(v_cache_start, v_cache_reorder);
          av_gemm_rest(
              attn_w,
              v_cache_reorder,
              attn_out_start,
              1);
        }
        if (complete_token_length == context_len) {
          // write to the output after the last token is done
          auto out_start = out_ptr +
            seq_id * attn_out_strideN + head_id * attn_out_strideH;
          torch_ipex::cpu::kernel::move_ker<scalar_t, float>(
              out_start, attn_out_start, head_size);
        }
      }
      // for the rest tail
      for (auto beam_id = 0; beam_id < beam_size; beam_id++) {
        auto seq_id = seq_id_start + beam_id;
        auto attn_out_start = fp32_attn_out_start +
            beam_id * fp32_attn_out_strideN;
        for (auto token_id = complete_token_length; token_id < context_len; token_id++) {
          auto attn_w = attn_weights_start
              [beam_id * attn_weights_strideN + token_id];
          auto block_id = block_tables_ptr
              [seq_id * max_num_blocks_per_seq + token_id / block_size];
          auto block_offset = token_id % block_size;
          auto v_cache_start = value_cache_ptr + block_id * kv_block_strideN +
              block_offset * kv_block_strideP +
              kv_head_id * kv_block_strideH;
          mul_attenion_weights_and_value_of_head<float, scalar_t>(
              attn_w,
              v_cache_start,
              attn_out_start,
              head_size,
              token_id);
        }
        // write to the output after the last token is done
        auto out_start = out_ptr +
          seq_id * attn_out_strideN + head_id * attn_out_strideH;
        torch_ipex::cpu::kernel::move_ker<scalar_t, float>(
            out_start, attn_out_start, head_size);
      }
    }
  }

} // single_query_cached_kv_attention_opt_kernel

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
 */
template <typename scalar_t>
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
    const c10::optional<at::Tensor>& alibi_slopes) {
  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
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
        float logits[16 * PARTITION_SIZE] __attribute__((aligned(64))) = {0};
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
 * store the key cache. The shape should be [num_blocks, num_key_value_heads
 * block_size, head_size].
 * @param value_cache The output value cache tensor. The pre-allocated buffer to
 * store the value cache. The shape should be [num_blocks, num_key_value_heads
 * block_size, head_size].
 * @param slot_mapping The slot mapping tensor. It stores the position to store
 * the key/value in the pre-allocated buffers. The shape should be the number of
 * sequences. For sequence i, the slot_mapping[i]//block_number can get the
 * block index, and the slot_mapping%block_size can get the offset of this
 * block.
 *
 * @tparam DST_T The data type of the output tensors.
 * @tparam SRC_T The data type of the input tensors.
 */
template <typename DST_T, typename SRC_T>
void reshape_and_cache_vnni_v_kernel(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping) {
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

  auto thread_numbers = omp_get_max_threads();
  auto v_xform = SCOPEIT(
      XformExtTPP<SRC_T>(
          /*in_rows*/ 2, //block_size,
          /*in_cols*/ head_size,
          /*out_rows*/ 2, //block_size,
          /*out_cols*/ head_size,
          /*ldi*/ cache_strideP,
          /*ldo*/ head_size,
          /*xtype*/ XformTPP::XFORM_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  int64_t size_per_thread =
    /* value_pad */ 2 * head_size +
    /* value_reorder */ 2 * head_size;

  at::Tensor total_buf = at::empty(
      {thread_numbers, size_per_thread},
      c10::CppTypeToScalarType<SRC_T>::value);
  auto total_buf_ptr = total_buf.data_ptr<SRC_T>();

#pragma omp parallel for collapse(2)
  for (auto ti = 0; ti < num_tokens; ti++) {
    for (auto hi = 0; hi < head_num; hi++) {
      auto ompIdx = omp_get_thread_num();
      SRC_T* buf = total_buf_ptr + ompIdx * size_per_thread;
      SRC_T* v_pad = buf;
      SRC_T* v_reorder = v_pad + 2 * head_size;
      auto physical_block_id = slot_mapping_ptr[ti] / block_size;
      auto block_offset = slot_mapping_ptr[ti] % block_size;
      auto cache_offset = physical_block_id * cache_strideN +
          block_offset * cache_strideP + hi * cache_strideH;
      auto curr_block_offset = block_offset % 2 ? block_offset - 1 : block_offset;
      auto v_cache_offset = physical_block_id * cache_strideN +
          curr_block_offset * cache_strideP + hi * cache_strideH;
      auto key_state_offset = ti * key_state_strideN + hi * key_state_strideH;
      auto value_state_offset =
          ti * value_state_strideN + hi * value_state_strideH;
      auto key_cache_start = key_cache_ptr + cache_offset;
      auto key_ptr_start = key_ptr + key_state_offset;
      auto value_cache_start = value_cache_ptr + v_cache_offset;
      auto value_ptr_start = value_ptr + value_state_offset;
      // for key
      torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(
            key_cache_start, key_ptr_start, head_size);
      // for value
      if (block_offset % 2 == 0) {
        // even case
        torch_ipex::cpu::kernel::move_ker<SRC_T, SRC_T>(
          v_pad, value_ptr_start, head_size);
        torch_ipex::cpu::kernel::zero_ker<SRC_T>(
          v_pad + head_size, head_size);
      } else {
        // odd case
        torch_ipex::cpu::kernel::move_ker<SRC_T, SRC_T>(
          v_pad + head_size, value_ptr_start, head_size);
        torch_ipex::cpu::kernel::zero_ker<SRC_T>(
          v_pad, head_size);
      }
      v_xform(v_pad, v_reorder);
      torch_ipex::cpu::kernel::mask_move_ker<DST_T, SRC_T>(
        value_cache_start, v_reorder, 2 * head_size);
    }
  }
}

/**
 * Reshapes and caches the key and value tensors based on the provided slot
 * mapping.
 *
 * @param key The input key tensor. The shape should be [num_seqs, num_heads,
 * head_size].
 * @param value The input value tensor.  The shape should be [num_seqs,
 * num_heads, head_size].
 * @param key_cache The output key cache tensor. The pre-allocated buffer to
 * store the key cache. The shape should be [num_blocks, num_key_value_heads
 * head_size, block_size].
 * @param value_cache The output value cache tensor. The pre-allocated buffer to
 * store the value cache. The shape should be [num_blocks, num_key_value_heads
 * block_size, head_size].
 * @param slot_mapping The slot mapping tensor. It stores the position to store
 * the key/value in the pre-allocated buffers. The shape should be the number of
 * sequences. For sequence i, the slot_mapping[i]//block_number can get the
 * block index, and the slot_mapping%block_size can get the offset of this
 * block.
 *
 * @tparam DST_T The data type of the output tensors.
 * @tparam SRC_T The data type of the input tensors.
 */
template <typename DST_T, typename SRC_T>
void reshape_and_cache_vnni_kernel(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping) {
  auto num_tokens = key.size(0);
  auto head_num = key.size(1);
  auto head_size = key.size(2);
  auto block_size = key_cache.size(3);
  auto hidden_size = head_num * head_size;
  auto key_cache_ptr = key_cache.data_ptr<DST_T>();
  auto key_ptr = key.data_ptr<SRC_T>();
  auto value_cache_ptr = value_cache.data_ptr<DST_T>();
  auto value_ptr = value.data_ptr<SRC_T>();
  auto slot_mapping_ptr = slot_mapping.data_ptr<int>();
  auto cache_strideN = key_cache.stride(0);
  auto cache_strideP = value_cache.stride(2);
  auto cache_strideS = key_cache.stride(2);
  auto cache_strideH = key_cache.stride(1);
  auto key_state_strideN = key.stride(0);
  auto key_state_strideH = key.stride(1);
  auto value_state_strideN = value.stride(0);
  auto value_state_strideH = value.stride(1);

  auto thread_numbers = omp_get_max_threads();
  auto k_xform = SCOPEIT(
      XformExtTPP<SRC_T>(
          /*in_rows*/ block_size,
          /*in_cols*/ head_size,
          /*out_rows*/ head_size,
          /*out_cols*/ block_size,
          /*ldi*/ head_size,
          /*ldo*/ block_size,
          /*xtype*/ XformTPP::XFORM_XPOSE_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  auto v_xform = SCOPEIT(
      XformExtTPP<SRC_T>(
          /*in_rows*/ 2, //block_size,
          /*in_cols*/ head_size,
          /*out_rows*/ 2, //block_size,
          /*out_cols*/ head_size,
          /*ldi*/ head_size,
          /*ldo*/ head_size,
          /*xtype*/ XformTPP::XFORM_N2V_TPP,
          /*ignore_vnni_for_fp32*/ true),
      XPOSE);

  int64_t size_per_thread =
    /* key_pad */ block_size * head_size +
    /* key_reorder */ head_size * block_size +
    /* value_pad */ 2 * head_size +
    /* value_reorder */ 2 * head_size;

  at::Tensor total_buf = at::empty(
      {thread_numbers, size_per_thread},
      c10::CppTypeToScalarType<SRC_T>::value);
  auto total_buf_ptr = total_buf.data_ptr<SRC_T>();

#pragma omp parallel for collapse(2)
  for (auto ti = 0; ti < num_tokens; ti++) {
    for (auto hi = 0; hi < head_num; hi++) {
      auto ompIdx = omp_get_thread_num();
      SRC_T* buf = total_buf_ptr + ompIdx * size_per_thread;
      SRC_T* k_pad = buf;
      SRC_T* k_reorder = k_pad + block_size * head_size;
      SRC_T* v_pad = k_reorder + head_size * block_size;
      SRC_T* v_reorder = v_pad + 2 * head_size;
      auto physical_block_id = slot_mapping_ptr[ti] / block_size;
      auto block_offset = slot_mapping_ptr[ti] % block_size;
      auto cache_offset = physical_block_id * cache_strideN +
          block_offset * cache_strideP + hi * cache_strideH;
      auto k_cache_offset = physical_block_id * cache_strideN +
          hi * cache_strideH;
      auto curr_block_offset = block_offset % 2 ? block_offset - 1 : block_offset;
      auto v_cache_offset = physical_block_id * cache_strideN +
          curr_block_offset * cache_strideP + hi * cache_strideH;
      auto key_state_offset =
          ti * key_state_strideN + hi * key_state_strideH;
      auto value_state_offset =
          ti * value_state_strideN + hi * value_state_strideH;
      auto key_cache_start = key_cache_ptr + k_cache_offset;
      auto key_ptr_start = key_ptr + key_state_offset;
      auto value_cache_start = value_cache_ptr + v_cache_offset;
      auto value_ptr_start = value_ptr + value_state_offset;
      // for key
      torch_ipex::cpu::kernel::zero_ker<SRC_T>(
          k_pad, block_size * head_size);
      torch_ipex::cpu::kernel::move_ker<SRC_T, SRC_T>(
          k_pad + block_offset * head_size, key_ptr_start, head_size);
      k_xform(k_pad, k_reorder);
      torch_ipex::cpu::kernel::mask_move_ker<DST_T, SRC_T>(
          key_cache_start, k_reorder, head_size * block_size);
      // for value
      if (block_offset % 2 == 0) {
        // even case
        torch_ipex::cpu::kernel::move_ker<SRC_T, SRC_T>(
          v_pad, value_ptr_start, head_size);
        torch_ipex::cpu::kernel::zero_ker<SRC_T>(
          v_pad + head_size, head_size);
      } else {
        // odd case
        torch_ipex::cpu::kernel::move_ker<SRC_T, SRC_T>(
          v_pad + head_size, value_ptr_start, head_size);
        torch_ipex::cpu::kernel::zero_ker<SRC_T>(
          v_pad, head_size);
      }
      v_xform(v_pad, v_reorder);
      torch_ipex::cpu::kernel::mask_move_ker<DST_T, SRC_T>(
        value_cache_start, v_reorder, 2 * head_size);
    }
  }
}


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
    at::Tensor& slot_mapping) {
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
      torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(
          key_cache_start, key_ptr_start, head_size);
      torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(
          value_cache_start, value_ptr_start, head_size);
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
template <typename scalar_t, int64_t q_split_size = 16>
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
    const c10::optional<at::Tensor>& alibi_slopes) {
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
  auto key_ptr = key_cache.data_ptr<scalar_t>();
  auto value_ptr = value_cache.data_ptr<scalar_t>();
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
    bool is_key_cache_vnni,
    at::Tensor& value_cache, // [num_blocks,  block_size, num_heads, head_size]
    bool is_value_cache_vnni,
    at::Tensor& head_mapping, // [num_heads]
    const double scale,
    at::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    at::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<at::Tensor>& alibi_slopes) {
  RECORD_FUNCTION(
      "ipex::single_query_cached_kv_attention_kernel_impl",
      c10::ArrayRef<c10::IValue>({}));
  // dispatch kernel according to the data type of input tensor
  auto head_size = query.size(2);
  if (head_size % 2 == 0 && block_size % 2 == 0) {
    if (is_key_cache_vnni && is_value_cache_vnni) {
      if (out.scalar_type() == at::ScalarType::Float) {
        single_query_cached_kv_attention_vnni_kernel<float>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else if (out.scalar_type() == at::ScalarType::BFloat16) {
        single_query_cached_kv_attention_vnni_kernel<at::BFloat16>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else if (out.scalar_type() == at::ScalarType::Half) {
        single_query_cached_kv_attention_vnni_kernel<at::Half>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else {
        TORCH_CHECK(
            false, "Unsupported data type for single_query_cached_kv_attention");
      }
    } else if (is_value_cache_vnni) {
      if (out.scalar_type() == at::ScalarType::Float) {
        single_query_cached_kv_attention_vnni_v_kernel<float>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else if (out.scalar_type() == at::ScalarType::BFloat16) {
        single_query_cached_kv_attention_vnni_v_kernel<at::BFloat16>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else if (out.scalar_type() == at::ScalarType::Half) {
        single_query_cached_kv_attention_vnni_v_kernel<at::Half>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else {
        TORCH_CHECK(
            false, "Unsupported data type for single_query_cached_kv_attention");
      }
    } else {
      if (out.scalar_type() == at::ScalarType::Float) {
        single_query_cached_kv_attention_opt_kernel<float>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else if (out.scalar_type() == at::ScalarType::BFloat16) {
        single_query_cached_kv_attention_opt_kernel<at::BFloat16>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else if (out.scalar_type() == at::ScalarType::Half) {
        single_query_cached_kv_attention_opt_kernel<at::Half>(
            out,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes);
      } else {
        TORCH_CHECK(
            false, "Unsupported data type for single_query_cached_kv_attention");
      }
    }
  } else {
    if (out.scalar_type() == at::ScalarType::Float) {
      single_query_cached_kv_attention_kernel<float>(
          out,
          query,
          key_cache,
          value_cache,
          scale,
          block_tables,
          context_lens,
          block_size,
          max_context_len,
          alibi_slopes);
    } else if (out.scalar_type() == at::ScalarType::BFloat16) {
      single_query_cached_kv_attention_kernel<at::BFloat16>(
          out,
          query,
          key_cache,
          value_cache,
          scale,
          block_tables,
          context_lens,
          block_size,
          max_context_len,
          alibi_slopes);
    } else if (out.scalar_type() == at::ScalarType::Half) {
      single_query_cached_kv_attention_kernel<at::Half>(
          out,
          query,
          key_cache,
          value_cache,
          scale,
          block_tables,
          context_lens,
          block_size,
          max_context_len,
          alibi_slopes);
    } else {
      TORCH_CHECK(
          false, "Unsupported data type for single_query_cached_kv_attention");
    }
  }
}

// void reshape_and_cache_kernel
void reshape_and_cache_cpu_kernel_impl(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    bool is_key_cache_vnni,
    at::Tensor& value_cache,
    bool is_value_cache_vnni,
    at::Tensor& slot_mapping) {
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
  if (is_key_cache_vnni && is_value_cache_vnni) {
    if (key.scalar_type() == at::ScalarType::Float) {
      reshape_and_cache_vnni_kernel<float, float>(
          key, value, key_cache, value_cache, slot_mapping);
    } else if (key.scalar_type() == at::ScalarType::BFloat16) {
      reshape_and_cache_vnni_kernel<at::BFloat16, at::BFloat16>(
          key, value, key_cache, value_cache, slot_mapping);
    } else if (key.scalar_type() == at::ScalarType::Half) {
      reshape_and_cache_vnni_kernel<at::Half, at::Half>(
          key, value, key_cache, value_cache, slot_mapping);
    } else {
      TORCH_CHECK(false, "Unsupported data type for ipex::reshape_and_cache");
    }
  } else if (is_value_cache_vnni) {
    if (key.scalar_type() == at::ScalarType::Float) {
      reshape_and_cache_vnni_v_kernel<float, float>(
          key, value, key_cache, value_cache, slot_mapping);
    } else if (key.scalar_type() == at::ScalarType::BFloat16) {
      reshape_and_cache_vnni_v_kernel<at::BFloat16, at::BFloat16>(
          key, value, key_cache, value_cache, slot_mapping);
    } else if (key.scalar_type() == at::ScalarType::Half) {
      reshape_and_cache_vnni_v_kernel<at::Half, at::Half>(
          key, value, key_cache, value_cache, slot_mapping);
    } else {
      TORCH_CHECK(false, "Unsupported data type for ipex::reshape_and_cache");
    }
  } else {
    if (key.scalar_type() == at::ScalarType::Float) {
      reshape_and_cache_kernel<float, float>(
          key, value, key_cache, value_cache, slot_mapping);
    } else if (key.scalar_type() == at::ScalarType::BFloat16) {
      reshape_and_cache_kernel<at::BFloat16, at::BFloat16>(
          key, value, key_cache, value_cache, slot_mapping);
    } else if (key.scalar_type() == at::ScalarType::Half) {
      reshape_and_cache_kernel<at::Half, at::Half>(
          key, value, key_cache, value_cache, slot_mapping);
    } else {
      TORCH_CHECK(false, "Unsupported data type for ipex::reshape_and_cache");
    }
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
    const c10::optional<at::Tensor>& alibi_slopes) {
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type(),
      "query and key should have the same data type");
  TORCH_CHECK(
      query.scalar_type() == value.scalar_type(),
      "query and value should have the same data type");
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
      flash_attn_varlen_kernel<float, 128>(
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
          alibi_slopes);
    } else if (max_seqlen_q >= 192) {
      flash_attn_varlen_kernel<float, 64>(
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
          alibi_slopes);
    } else {
      flash_attn_varlen_kernel<float, 32>(
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
          alibi_slopes);
    }

  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    if (max_seqlen_q >= 768) {
      flash_attn_varlen_kernel<at::BFloat16, 128>(
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
          alibi_slopes);
    } else if (max_seqlen_q >= 192) {
      flash_attn_varlen_kernel<at::BFloat16, 64>(
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
          alibi_slopes);
    } else {
      flash_attn_varlen_kernel<at::BFloat16, 32>(
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
          alibi_slopes);
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
