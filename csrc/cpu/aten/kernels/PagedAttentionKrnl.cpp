#include <ATen/Tensor.h>
#include <aten/PagedAttention.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

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
 * @param head_mapping  Head mapping tensor [num_heads]. The mapping from the
 * query head to the kv head to support GQA/MQA. The shape should be the number
 * of query heads.
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
    at::Tensor& head_mapping,
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
  auto head_mapping_ptr = head_mapping.data_ptr<int>();
  auto block_tables_ptr = block_tables.data_ptr<int>();
  auto context_lens_ptr = context_lens.data_ptr<int>();
  auto alibi_slopes_ptr = alibi_slopes.has_value()
      ? alibi_slopes.value().data_ptr<float>()
      : nullptr;
  auto num_seqs = query.size(0);
  auto num_heads = query.size(1);
  auto head_size = query.size(2);
  auto num_kv_heads = key_cache.size(2);
  auto max_num_blocks_per_seq = block_tables.size(1);
  auto attn_weights = at::empty(
      {num_seqs, num_heads, max_context_len},
      query.options().dtype(at::ScalarType::Float));
  auto attn_weights_ptr = attn_weights.data_ptr<float>();
  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = key_cache.stride(1);
  auto kv_block_strideH = key_cache.stride(2);

  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);
  auto attn_weights_strideN = attn_weights.stride(0);
  auto attn_weights_strideH = attn_weights.stride(1);

  auto max_logic_blocks = (max_context_len + block_size - 1) / block_size;

  auto thread_numbers = omp_get_max_threads();
  auto max_parallel_parts = thread_numbers * 4;
  if (alibi_slopes.has_value()) {
    auto alibi_slopes_size = alibi_slopes.value().size(0);
    TORCH_CHECK(
        alibi_slopes_size == num_heads,
        "alibi_slopes size is not equal to num_heads");
  }
  {
    RECORD_FUNCTION(
        "ipex::paged_attention_sdp::matmul(query, key)",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3) schedule(static, 1)
    for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
      for (auto logical_block_id = 0; logical_block_id < max_logic_blocks;
           logical_block_id++) {
        for (auto head_id = 0; head_id < num_heads; head_id++) {
          auto context_len = context_lens_ptr[seq_id];
          auto token_start = logical_block_id * block_size;
          auto token_end =
              std::min((int)(token_start + block_size), context_len);
          if (token_start >= context_len)
            continue;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto attn_w_pos = attn_weights_ptr + seq_id * attn_weights_strideN +
                head_id * attn_weights_strideH + token_id;
            auto q_ptr_start =
                query_ptr + seq_id * q_strideN + head_id * q_strideH;
            auto physical_block_id = block_tables_ptr
                [seq_id * max_num_blocks_per_seq + logical_block_id];
            auto block_offset = token_id - token_start;
            auto k_cache_start = key_cache_ptr +
                physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP +
                head_mapping_ptr[head_id] * kv_block_strideH;
            reduce_head<scalar_t, scalar_t>(
                q_ptr_start, k_cache_start, attn_w_pos, head_size);
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::paged_attention_sdp::div+add+softmax",
        c10::ArrayRef<c10::IValue>({}));
// div+add+softmax
#pragma omp parallel for collapse(2)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto max_val = -10000.0f;
      float sum = 0.0f;
      auto context_len = context_lens_ptr[seq_id];
      auto attn_w_start = attn_weights_ptr + seq_id * attn_weights_strideN +
          head_id * attn_weights_strideH;
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
      torch_ipex::cpu::kernel::_dil_normalization_kernel<float>(
          attn_w_start, max_val, context_len, attn_w_start);

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
      for (auto token_id = 0; token_id < context_lens_ptr[seq_id]; token_id++) {
        attn_w_start[token_id] = attn_w_start[token_id] / sum;
      }
#endif
    }
  }
  }
  auto private_attn_outs =
      at::empty({thread_numbers, num_seqs, num_heads, head_size}, at::kFloat);
  auto private_attn_out_flag =
      at::zeros({thread_numbers, num_seqs, num_heads}, at::kByte);
  auto flag_access = private_attn_out_flag.accessor<uint8_t, 3>();
  auto private_attn_out_ptr = private_attn_outs.data_ptr<float>();
  auto private_attn_out_strideT = private_attn_outs.stride(0);
  auto private_attn_out_strideN = private_attn_outs.stride(1);
  auto private_attn_out_strideH = private_attn_outs.stride(2);
  auto attn_out_strideN = out.stride(0);
  auto attn_out_strideH = out.stride(1);
  {
    RECORD_FUNCTION(
        "ipex::paged_attention_sdp::matmul(attn_w, value)",
        c10::ArrayRef<c10::IValue>({}));
// mul and accumulate
#pragma omp parallel for collapse(3) schedule(static, 1)
    for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
      for (auto logical_block_id = 0; logical_block_id < max_logic_blocks;
           logical_block_id++) {
        for (auto head_id = 0; head_id < num_heads; head_id++) {
          auto context_len = context_lens_ptr[seq_id];
          auto token_start = logical_block_id * block_size;
          if (token_start >= context_len)
            continue;
          auto token_end =
              std::min((int)(token_start + block_size), context_len);
          auto thread_id = omp_get_thread_num();
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto attn_w = attn_weights_ptr
                [seq_id * attn_weights_strideN +
                 head_id * attn_weights_strideH + token_id];
            auto physical_block_id = block_tables_ptr
                [seq_id * max_num_blocks_per_seq + logical_block_id];
            auto block_offset = token_id - token_start;
            auto v_cache_start = value_cache_ptr +
                physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP +
                head_mapping_ptr[head_id] * kv_block_strideH;
            auto attn_out_start = private_attn_out_ptr +
                thread_id * private_attn_out_strideT +
                seq_id * attn_out_strideN + head_id * attn_out_strideH;
            mul_attenion_weights_and_value_of_head<float, scalar_t>(
                attn_w,
                v_cache_start,
                attn_out_start,
                head_size,
                flag_access[thread_id][seq_id][head_id]);
            if (flag_access[thread_id][seq_id][head_id] == 0) {
              flag_access[thread_id][seq_id][head_id] = 1;
            }
          } // for token_id
        } // for head_id
      } // for block id
    } // for seq_id
  }
  {
    RECORD_FUNCTION(
        "ipex::single_query_cached_kv_attention::reduction_private_result",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2)
    for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
      for (auto hi = 0; hi < num_heads; hi++) {
        auto thr0_head_start = private_attn_out_ptr +
            seq_id * private_attn_out_strideN + hi * private_attn_out_strideH;
        if (flag_access[0][seq_id][hi] == 0) {
          torch_ipex::cpu::kernel::zero_ker(thr0_head_start, head_size);
        }
        for (auto thread_id = 1; thread_id < thread_numbers; thread_id++) {
          if (flag_access[thread_id][seq_id][hi] == 0) {
            continue;
          }
          auto attn_out_head_offset = thread_id * private_attn_out_strideT +
              seq_id * private_attn_out_strideN + hi * private_attn_out_strideH;
          auto private_attn_out_start =
              private_attn_out_ptr + attn_out_head_offset;
          torch_ipex::cpu::kernel::add_ker<float, float>(
              thr0_head_start, private_attn_out_start, head_size);
        }
        auto out_start = out_ptr + (seq_id * num_heads + hi) * head_size;
        torch_ipex::cpu::kernel::move_ker<scalar_t, float>(
            out_start, thr0_head_start, head_size);
      }
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
  auto block_size = key_cache.size(1);
  auto hidden_size = head_num * head_size;
  auto key_cache_ptr = key_cache.data_ptr<DST_T>();
  auto key_ptr = key.data_ptr<SRC_T>();
  auto value_cache_ptr = value_cache.data_ptr<DST_T>();
  auto value_ptr = value.data_ptr<SRC_T>();
  auto slot_mapping_ptr = slot_mapping.data_ptr<int>();
  auto cache_strideN = key_cache.stride(0);
  auto cache_strideP = key_cache.stride(1);
  auto cache_strideH = key_cache.stride(2);
  auto state_strideN = key.stride(0);
  auto state_strideH = key.stride(1);
#pragma omp parallel for collapse(2)
  for (auto ti = 0; ti < num_tokens; ti++) {
    for (auto hi = 0; hi < head_num; hi++) {
      auto physical_block_id = slot_mapping_ptr[ti] / block_size;
      auto block_offset = slot_mapping_ptr[ti] % block_size;
      auto cache_offset = physical_block_id * cache_strideN +
          block_offset * cache_strideP + hi * cache_strideH;
      auto state_offset = ti * state_strideN + hi * state_strideH;
      auto key_cache_start = key_cache_ptr + cache_offset;
      auto key_ptr_start = key_ptr + state_offset;
      auto value_cache_start = value_cache_ptr + cache_offset;
      auto value_ptr_start = value_ptr + state_offset;
      torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(
          key_cache_start, key_ptr_start, head_size);
      torch_ipex::cpu::kernel::move_ker<DST_T, SRC_T>(
          value_cache_start, value_ptr_start, head_size);
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
    const c10::optional<at::Tensor>& alibi_slopes) {
  RECORD_FUNCTION(
      "ipex::single_query_cached_kv_attention_kernel_impl",
      c10::ArrayRef<c10::IValue>({}));
  // dispatch kernel according to the data type of input tensor
  if (out.scalar_type() == at::ScalarType::Float) {
    single_query_cached_kv_attention_kernel<float>(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
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
        head_mapping,
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

// void reshape_and_cache_kernel
void reshape_and_cache_cpu_kernel_impl(
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
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
  if (key.scalar_type() == at::ScalarType::Float) {
    reshape_and_cache_kernel<float, float>(
        key, value, key_cache, value_cache, slot_mapping);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    reshape_and_cache_kernel<at::BFloat16, at::BFloat16>(
        key, value, key_cache, value_cache, slot_mapping);
  } else {
    TORCH_CHECK(false, "Unsupported data type for ipex::reshape_and_cache");
  }
}

} // namespace

IPEX_REGISTER_DISPATCH(
    single_query_cached_kv_attention_kernel_stub,
    &single_query_cached_kv_attention_kernel_impl);
IPEX_REGISTER_DISPATCH(
    reshape_and_cache_kernel_stub,
    &reshape_and_cache_cpu_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
