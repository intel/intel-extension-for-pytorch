/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in wriscalar_tg, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

// limitation: as 2d load/store instruction requires surface width
// (encoded_value + 1) must be a multiple of DW (4 bytes).
//  seqlen *  sizeof(data) % 4 == 0, meaning seqlen % 2 = 0

#pragma once

#include <sys/types.h>
#include "blocked_attention_utils.hpp"
#include "common/core/common.hpp"
#include "common/core/common_types.hpp"
#include "common/core/memory.hpp"
#include "common/utils/misc.hpp"
#include "fmha_utils.h"
#include "paged_attention_utils.hpp"
#include "xetla.hpp"

#include <sycl/sycl.hpp>

namespace gpu::xetla {

namespace attention {

#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

template <typename policy, typename scalar_t, gpu_arch arch_tag>
class blocked_attention_kernel {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input and output tensors
    scalar_t* out; // [num_tokens, num_heads, max_num_partitions, head_size]
    accum_t* max_logits; // [num_tokens, num_heads, max_num_partitions]
    accum_t* exp_sums; // [num_tokens, num_heads, max_num_partitions]
    scalar_t* query; // [num_tokens, num_heads, head_size]
    scalar_t* key_cache; // [num_blocks, block_size, num_kv_heads, head_size]
    scalar_t* value_cache; // [num_blocks, block_size, num_kv_heads, head_size]

    int32_t* block_tables; // [num_seqs, max_blocks_per_seq]
    int32_t* cu_seqlen_q; // [num_seqs + 1]
    int32_t* cu_seqlen_k; // [num_seqs + 1]
    // AttentionAtom* atoms; // [num_atoms]

    // Softmax scale
    accum_t sm_scale;

    // Stride size
    uint32_t q_row_stride;
    uint32_t kv_row_stride;

    // Dimension size
    uint32_t num_atoms;
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t max_queries;
    uint32_t max_keys;
    uint32_t head_groups;
    uint32_t head_size;
    uint32_t max_blocks_per_seq;
    uint32_t max_num_partitions;
    bool is_causal;

    inline arguments_t(
        scalar_t* out,
        accum_t* max_logits,
        accum_t* exp_sums,
        scalar_t* query,
        scalar_t* key_cache,
        scalar_t* value_cache,
        // AttentionAtom* atoms_,
        int32_t* block_tables,
        int32_t* cu_seqlen_q,
        int32_t* cu_seqlen_k,
        accum_t sm_scale,
        uint32_t q_row_stride,
        uint32_t kv_row_stride,
        // uint32_t num_atoms,
        uint32_t max_queries,
        uint32_t max_keys,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t max_blocks_per_seq,
        uint32_t max_num_partitions,
        bool is_causal)
        : out(out),
          max_logits(max_logits),
          exp_sums(exp_sums),
          query(query),
          key_cache(key_cache),
          value_cache(value_cache),
          block_tables(block_tables),
          cu_seqlen_q(cu_seqlen_q),
          cu_seqlen_k(cu_seqlen_k),
          sm_scale(sm_scale),
          q_row_stride(q_row_stride),
          kv_row_stride(kv_row_stride),
          // num_atoms(num_atoms),
          max_queries(max_queries),
          max_keys(max_keys),
          batch_size(batch_size),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_size(head_size),
          max_blocks_per_seq(max_blocks_per_seq),
          max_num_partitions(max_num_partitions),
          is_causal(is_causal) {
      // atoms = atoms_; // reinterpret_cast<AttentionAtom*>(atoms_);
      head_groups = num_heads / num_kv_heads;
    }
  };

 private:
  // -------------------- // Compute policy // -------------------- //

  static constexpr accum_t neg_infinity = INFINITY * -1;
  static constexpr uint32_t stages = policy::stages;
  static constexpr uint32_t wg_size = policy::wg_size;
  static constexpr uint32_t block_size = policy::block_size;
  static constexpr uint32_t block_size_y = 8;
  static constexpr uint32_t wg_size_y = 1;
  static constexpr uint32_t max_head_size = policy::max_head_size;
  static constexpr uint32_t head_size_stride = policy::head_size_stride;
  static constexpr uint32_t max_blocks_per_sg = policy::max_blocks_per_sg;
  // used for preload query and store output
  // use minimum 16 to avoid mask in load
  static constexpr uint32_t head_size_per_sg =
      std::max(max_head_size / wg_size, 16u);

  // -------------------- // Slm and nbarrier // -------------------- //

  // each atom contains q_block_size tokens
  static constexpr uint32_t slm_size_query =
      block_size_y * max_head_size * sizeof(scalar_t);
  static constexpr uint32_t slm_size_softmax =
      (wg_size > 1) ? block_size_y * wg_size * sizeof(accum_t) : 0;
  static constexpr uint32_t slm_size_out =
      max_head_size * wg_size * sizeof(accum_t);

  static constexpr uint32_t slm_offset_query = 0;
  static constexpr uint32_t slm_offset_softmax =
      slm_offset_query + slm_size_query;
  static constexpr uint32_t slm_offset_out =
      slm_offset_softmax + slm_size_softmax;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? wg_size_y : 0;

  // -------------------- // Context // -------------------- //

  struct context_t {
    uint32_t sg_id_x;
    uint32_t wg_id;
    uint32_t batch_id;
    uint32_t head_id;
    uint32_t group_id;
    uint32_t partition_id;
    uint32_t num_partitions;
    uint32_t max_blocks_per_wg;

    uint32_t seq_q_start;
    uint32_t seq_q_end;
    uint32_t seq_k_start;
    uint32_t seq_k_end;

    // AttentionAtom* atom_info;
    int32_t* block_table;

    int start_block_id;
    int end_block_id;
    int loop_count;

    xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;

    inline context_t() = default;

    inline void init(sycl::nd_item<3>& item, arguments_t& args) {
      sg_id_x = item.get_local_id(2);
      // sg_id_y = item.get_local_id(1);
      head_id = item.get_group(0) % args.num_heads;
      batch_id = item.get_group(0) / args.num_heads;
      group_id = item.get_group(1);
      partition_id = item.get_group(2);
      num_partitions = item.get_group_range(2);

      seq_q_start = args.cu_seqlen_q[batch_id];
      seq_q_end = args.cu_seqlen_q[batch_id + 1];
      seq_k_start = args.cu_seqlen_k[batch_id];
      seq_k_end = args.cu_seqlen_k[batch_id + 1];
      max_blocks_per_wg =
          DIVIDE_ROUND_UP(args.max_blocks_per_seq, num_partitions);
      block_table = args.block_tables + batch_id * args.max_blocks_per_seq +
          partition_id * max_blocks_per_wg;

      loop_count = DIVIDE_ROUND_UP(args.head_size, head_size_stride);

      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
    }
  };

  context_t ctx;

  inline bool thread0() {
    return ctx.sg_id_x == 0 && ctx.batch_id == 0 && ctx.group_id == 0;
  }
  // -------------------- // preload_query // -------------------- //

  // Pre-load query from global memory to shared local memory.
  inline void preload_query(arguments_t& args) {
    // TODO: update policy for short query len
    using tile_desc_t =
        subgroup::tile_desc_t<head_size_per_sg, block_size_y, 16, block_size_y>;
    using tile_t = subgroup::tile_t<scalar_t, tile_desc_t>;

    using global_ld_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    using local_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
        tile_desc_t,
        msg_type::scatter,
        arch_tag>;

    // get [block_size_y, head_size_per_sg] of Q
    int32_t start_x_local = ctx.sg_id_x * head_size_per_sg;
    int32_t start_x = start_x_local;
    int32_t end_x = start_x + head_size_per_sg;
    end_x = end_x < args.head_size ? end_x : args.head_size;
    // int32_t start_y = ctx.atom_info->q_start_idx;
    int32_t start_y = ctx.seq_q_start + ctx.group_id * block_size_y;
    int32_t end_y = start_y + block_size_y;
    int32_t limit_y = ctx.seq_q_end;
    end_y = end_y < limit_y ? end_y : limit_y;
    int32_t pitch = args.q_row_stride;
    auto query_ptr = args.query + ctx.head_id * args.head_size;
    global_ld_payload_t ld_payload(
        query_ptr, end_x, end_y, pitch, start_x, start_y);

    local_st_payload_t st_payload(
        slm_offset_query, end_x, block_size_y, max_head_size, start_x_local, 0);
    tile_t mat_query;

    subgroup::tile_load(mat_query, ld_payload);

    subgroup::tile_store(mat_query, st_payload);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

  // -------------------- // compute_score // -------------------- //

  static constexpr uint32_t mma_sg_tile_size = 16;
  using score_tile_desc_t = subgroup::tile_desc_t<
      block_size * max_blocks_per_sg,
      block_size_y,
      mma_sg_tile_size,
      block_size_y>;
  using score_tile_t = subgroup::tile_t<accum_t, score_tile_desc_t>;

  // Compute query x key.
  inline void compute_score(score_tile_t& mat_score, arguments_t& args) {
    constexpr uint32_t sg_tile_size_k = head_size_stride > 32 / sizeof(scalar_t)
        ? 32 / sizeof(scalar_t)
        : head_size_stride;
    constexpr uint32_t sg_tile_size_n = mma_sg_tile_size;
    constexpr uint32_t sg_tile_size_m = block_size_y > 32 ? 32 : block_size_y;
    using score_acc_tile_desc_t = subgroup::
        tile_desc_t<block_size, block_size_y, sg_tile_size_n, sg_tile_size_m>;
    using score_acc_tile_t = subgroup::tile_t<accum_t, score_acc_tile_desc_t>;
    using query_tile_desc_t = subgroup::tile_desc_t<
        head_size_stride,
        block_size_y,
        sg_tile_size_k,
        sg_tile_size_m>;
    using query_tile_t = subgroup::tile_t<scalar_t, query_tile_desc_t>;
    using query_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
        query_tile_desc_t,
        msg_type::scatter,
        arch_tag>;
    constexpr tdesc_update_dir query_update_dir = tdesc_update_dir::x_dir;

    using key_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        head_size_stride,
        sg_tile_size_n,
        sg_tile_size_k,
        reg_layout::vnni_tiled>;

    constexpr mem_layout k_mem_layout = mem_layout::col_major;
    constexpr tdesc_update_dir key_update_dir = tdesc_update_dir::x_dir;
    using key_tile_t = subgroup::tile_t<scalar_t, key_tile_desc_t>;
    using key_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, k_mem_layout, mem_space::global>,
        key_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    using key_prefetch_payload_t = subgroup::prefetch_payload_t<
        mem_desc_t<scalar_t, k_mem_layout, mem_space::global>,
        key_tile_desc_t,
        1,
        arch_tag>;
    using tile_mma = subgroup::tile_mma_t<
        score_acc_tile_t,
        score_acc_tile_t,
        key_tile_t,
        query_tile_t,
        mma_engine::xmx,
        arch_tag>;

    query_tile_t mat_query;
    key_tile_t mat_key;

    uint32_t seqlen_q = ctx.seq_q_end - ctx.seq_q_start;
    uint32_t seqlen_k = ctx.seq_k_end - ctx.seq_k_start;
    uint32_t seqlen_diff = seqlen_k - seqlen_q;
    uint32_t context_q_start = seqlen_diff + ctx.group_id * block_size_y;
    uint32_t context_q_end = context_q_start + block_size_y > seqlen_k
        ? seqlen_k
        : context_q_start + block_size_y;

    // iterate over context blocks
    for (int bid = ctx.sg_id_x, row_i = 0; bid < ctx.max_blocks_per_wg;
         bid += wg_size, row_i++) {
      // get the physical block id from block_table
      const int block_id = ctx.block_table[bid];
      const int cu_seqlen = bid * block_size + ctx.seq_k_start +
          ctx.partition_id * ctx.max_blocks_per_wg * block_size;
      if (cu_seqlen >= ctx.seq_k_end) {
        break;
      }
      const int context_k_start = bid * block_size +
          ctx.partition_id * ctx.max_blocks_per_wg * block_size;

      if (args.is_causal && context_k_start >= context_q_end) {
        break;
      }

      // load [block_size, head_size] for each sg, but will fill it within
      // loop_count times actually get transposed [head_size, block_size]
      int32_t start_x = ctx.head_id / args.head_groups * args.head_size;
      int32_t end_x = start_x + args.head_size;
      int32_t start_y = block_id * block_size;
      int32_t end_y = start_y + block_size;
      key_payload_t key_payload(
          args.key_cache, end_x, end_y, args.kv_row_stride, start_x, start_y);
      key_prefetch_payload_t key_prefetch_payload(
          args.key_cache, end_x, end_y, args.kv_row_stride, start_x, start_y);
      query_payload_t query_payload(
          slm_offset_query, max_head_size, block_size_y, max_head_size, 0, 0);
#if 0
#pragma unroll
      for (int i = 0; i < stages; i++) {
        subgroup::tile_prefetch(key_prefetch_payload);
        key_prefetch_payload.template update_tdesc<key_update_dir>(
            head_size_stride);
      }
#endif

      score_acc_tile_t score_sub(0.0f);

      for (int i = 0; i < ctx.loop_count; i++) {
        subgroup::tile_load(mat_query, query_payload);
        subgroup::tile_load(mat_key, key_payload);
#if 0
        if constexpr (stages != 0) {
          subgroup::tile_prefetch(key_prefetch_payload);
        }
#endif
        // SW_BARRIER();
        query_payload.template update_tdesc<query_update_dir>(head_size_stride);
        key_payload.template update_tdesc<key_update_dir>(head_size_stride);
#if 0
        if constexpr (stages != 0) {
          key_prefetch_payload.template update_tdesc<key_update_dir>(
              head_size_stride);
        }
#endif
        // #if 0
        // SW_BARRIER();
        tile_mma::mma(score_sub, score_sub, mat_key, mat_query);
        // SW_BARRIER();
        // #endif
      }

      score_sub.reg *= args.sm_scale;
      // TODO: handle remained_len before the loop and reverse bid loope
      using tile_mask = fmha::tile_mask_t<score_acc_tile_t>;
      uint32_t remained_len = std::max(int(ctx.seq_k_end) - int(cu_seqlen), 0);

      if (remained_len < block_size) {
        tile_mask::padding_mask(score_sub, remained_len);
      }
      // TODO: handle remained_len before the loop and reverse bid loop
      if (args.is_causal && context_k_start + block_size > context_q_start) {
        tile_mask::causal_mask(score_sub, context_k_start, context_q_start);
      }

      mat_score.reg.xetla_select<block_size_y * block_size, 1>(
          row_i * block_size_y * block_size) = score_sub.reg;
    }
  }

  // -------------------- // softmax // -------------------- //

  // Compute softmax of score.
  inline void softmax(score_tile_t& mat_score, arguments_t& args) {
    using wg_row_max_t = fmha::
        group_row_reduce_t<score_tile_t, wg_size, reduce_op::max, arch_tag>;
    using wg_row_sum_t = fmha::
        group_row_reduce_t<score_tile_t, wg_size, reduce_op::sum, arch_tag>;
    using global_st_softmax_tile_desc_t =
        subgroup::tile_desc_t<1, block_size_y, 1, block_size_y>;
    using global_st_softmax_tile_t =
        subgroup::tile_t<accum_t, global_st_softmax_tile_desc_t>;
    using global_st_softmax_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        global_st_softmax_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    wg_row_max_t wg_row_max(ctx.sg_id_x, 0, slm_offset_softmax);
    xetla_vector<accum_t, block_size_y> group_max = wg_row_max(mat_score);
    xetla_mask<block_size_y> mask_inf = group_max == neg_infinity;
    xetla_vector<accum_t, block_size_y> group_max_minus = group_max;
    group_max_minus.xetla_merge(0.0f, mask_inf);

    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive();

    subgroup::tile_broadcast_op<subgroup::tile_minus, score_tile_t>(
        mat_score, group_max_minus);
    mat_score.reg = xetla_exp<accum_t>(mat_score.reg);

    if constexpr (wg_size > 1)
      ctx.nbarrier.wait();

    wg_row_sum_t wg_row_sum(ctx.sg_id_x, 0, slm_offset_softmax);
    xetla_vector<accum_t, block_size_y> group_sum = wg_row_sum(mat_score);

    subgroup::tile_broadcast_op<subgroup::tile_div, score_tile_t>(
        mat_score, group_sum + std::numeric_limits<accum_t>::min());
    // #endif
    // sycl::ext::oneapi::experimental::printf("num partitions: %d\n",
    // ctx.num_partitions); int num_pars = ctx.num_partitions; if the kv is too
    // larget, we adopt split kv to maximize the occupancy, and save the max
    // value and exp value to global mem for another reduction.
    if (args.max_num_partitions > 1) {
      /* The shape of max_logits and exp_sums are
          [num_tokens, num_heads, max_num_partitions]
      */
      global_st_softmax_tile_t max_logits_tile(0.0f);
      global_st_softmax_tile_t exp_sums_tile(0.0f);
      max_logits_tile.reg = group_max;
      exp_sums_tile.reg = group_sum;
      int32_t start_x = ctx.partition_id;
      int32_t start_y = ctx.seq_q_start + ctx.group_id * block_size_y;
      int32_t end_x = start_x + 1;
      end_x = end_x > args.max_num_partitions ? args.max_num_partitions : end_x;
      int32_t end_y = start_y + block_size_y;
      int32_t boundary_y = ctx.seq_q_end;
      end_y = end_y < boundary_y ? end_y : boundary_y;

      int32_t ptr_offset = ctx.head_id * args.max_num_partitions;
      auto max_logits_ptr = args.max_logits + ptr_offset;
      auto exp_sums_ptr = args.exp_sums + ptr_offset;
      global_st_softmax_payload_t softmax_st_payload(
          max_logits_ptr,
          end_x,
          end_y,
          args.max_num_partitions * args.num_heads,
          start_x,
          start_y);
      subgroup::tile_store(max_logits_tile, softmax_st_payload);
      global_st_softmax_payload_t exp_sums_st_payload(
          exp_sums_ptr,
          end_x,
          end_y,
          args.max_num_partitions * args.num_heads,
          start_x,
          start_y);
      subgroup::tile_store(exp_sums_tile, exp_sums_st_payload);
    }
  }

  // -------------------- // comput_out // -------------------- //

  using out_tile_desc_t = subgroup::
      tile_desc_t<max_head_size, block_size_y, mma_sg_tile_size, block_size_y>;
  using out_tile_t = subgroup::tile_t<accum_t, out_tile_desc_t>;

  // Compute output.
  inline void compute_out(
      score_tile_t& mat_score,
      out_tile_t& mat_out,
      arguments_t& args) {
    constexpr uint32_t sg_tile_size_k =
        block_size > 32 / sizeof(scalar_t) ? 32 / sizeof(scalar_t) : block_size;
    constexpr uint32_t sg_tile_size_n = mma_sg_tile_size;
    constexpr uint32_t sg_tile_size_m = block_size_y > 32 ? 32 : block_size_y;
    using out_acc_tile_desc_t = subgroup::tile_desc_t<
        head_size_stride,
        block_size_y,
        sg_tile_size_n,
        sg_tile_size_m>;
    using out_acc_tile_t = subgroup::tile_t<accum_t, out_acc_tile_desc_t>;
    using score_acc_tile_desc_t = subgroup::
        tile_desc_t<block_size, block_size_y, sg_tile_size_k, sg_tile_size_m>;
    using score_acc_tile_t = subgroup::tile_t<scalar_t, score_acc_tile_desc_t>;
    using value_tile_desc_t = subgroup::tile_desc_t<
        head_size_stride,
        block_size,
        sg_tile_size_n,
        sg_tile_size_k,
        reg_layout::vnni_tiled>;
    using value_tile_t = subgroup::tile_t<scalar_t, value_tile_desc_t>;
    // using value_acc_tile_t = subgroup::tile_t<accum_t, value_tile_desc_t>;
    using value_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        value_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    using value_prefetch_payload_t = subgroup::prefetch_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        value_tile_desc_t,
        1,
        arch_tag>;
    constexpr tdesc_update_dir value_update_dir = tdesc_update_dir::x_dir;
    using tile_mma = subgroup::tile_mma_t<
        out_acc_tile_t,
        out_acc_tile_t,
        value_tile_t,
        score_acc_tile_t,
        mma_engine::xmx,
        arch_tag>;

    value_tile_t mat_value;

    // iterate over context blocks
    for (int bid = ctx.sg_id_x, row_i = 0; bid < ctx.max_blocks_per_wg;
         bid += wg_size, row_i++) {
      // get the physical block id from block_table
      const int block_id = ctx.block_table[bid];
      const int cu_seqlen = bid * block_size + ctx.seq_k_start +
          ctx.partition_id * ctx.max_blocks_per_wg * block_size;
      if (cu_seqlen >= ctx.seq_k_end) {
        break;
      }

      // load [block_size, head_size]
      int32_t start_x = ctx.head_id / args.head_groups * args.head_size;
      int32_t end_x = start_x + args.head_size;
      int32_t start_y = block_id * block_size;
      int32_t remained_y = ctx.seq_k_end - cu_seqlen;
      int32_t boundary_y = remained_y < block_size ? remained_y : block_size;
      int32_t end_y = start_y + boundary_y;
      value_payload_t value_payload(
          args.value_cache, end_x, end_y, args.kv_row_stride, start_x, start_y);
      value_prefetch_payload_t value_prefetch_payload(
          args.value_cache, end_x, end_y, args.kv_row_stride, start_x, start_y);

#if 0
#pragma unroll
      for (int i = 0; i < stages; i++) {
        subgroup::tile_prefetch(value_prefetch_payload);
        value_prefetch_payload.template update_tdesc<value_update_dir>(
            head_size_stride);
      }
#endif
      score_acc_tile_t score_sub;
      score_sub.reg = xetla_cvt<scalar_t, accum_t, block_size_y * block_size>(
          mat_score.reg.xetla_select<block_size_y * block_size, 1>(
              row_i * block_size * block_size_y));

      for (int i = 0; i < ctx.loop_count; i++) {
        subgroup::tile_load(mat_value, value_payload);
#if 0
        if constexpr (stages != 0) {
          subgroup::tile_prefetch(value_prefetch_payload);
        }
#endif
        // SW_BARRIER();
        value_payload.template update_tdesc<value_update_dir>(head_size_stride);
#if 0
        if constexpr (stages != 0) {
          value_prefetch_payload.template update_tdesc<value_update_dir>(
              head_size_stride);
        }
#endif
        // SW_BARRIER();
        out_acc_tile_t out_sub;
        out_sub.reg =
            mat_out.reg.xetla_select<block_size_y * head_size_stride, 1>(
                i * block_size_y * head_size_stride);
        tile_mma::mma(out_sub, out_sub, mat_value, score_sub);
        mat_out.reg.xetla_select<block_size_y * head_size_stride, 1>(
            i * block_size_y * head_size_stride) = out_sub.reg;
        // SW_BARRIER();
      }
    }
  }

  // -------------------- // collect_out // -------------------- //

  inline void collect_out(out_tile_t& mat_out, arguments_t& args) {
    // 1. store [block_size_y, max_head_size] to slm
    // 2. reduce within slm, each subgroup calc [block_size_y, wg_size,
    // head_size_per_sg]
    // 3. each subgroup write out [block_size_y, head_size_per_sg] to global

    constexpr uint32_t sg_tile_size_x = head_size_per_sg > 32 / sizeof(scalar_t)
        ? 32 / sizeof(scalar_t)
        : head_size_per_sg;
    constexpr uint32_t sg_tile_size_y = wg_size > 16 ? 16 : wg_size;
    int32_t start_x_local = ctx.sg_id_x * head_size_per_sg;

    // for storing out to global
    using st_tile_desc_t = subgroup::tile_desc_t<
        head_size_per_sg,
        block_size_y,
        sg_tile_size_x,
        block_size_y>;
    using global_st_tile_t = subgroup::tile_t<scalar_t, st_tile_desc_t>;
    using global_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        st_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    global_st_tile_t mat_out_st;
    uint32_t seq_len_q = ctx.seq_q_end - ctx.seq_q_start;
    for (uint32_t i = 0; i < block_size_y; i++) {
      // 1. store out data of each subgroup to slm
      // [block_size_y, wg_size, max_head_size]
      // whole mat_out is too big to store in slm, so we store [wg_size,
      // max_head_size]
      using local_st_tile_desc_t =
          subgroup::tile_desc_t<max_head_size, 1, mma_sg_tile_size, 1>;
      using local_st_tile_t = subgroup::tile_t<accum_t, local_st_tile_desc_t>;
      using local_st_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
          local_st_tile_desc_t,
          msg_type::scatter,
          arch_tag>;
      local_st_tile_t mat_sub_out;
      // need multiple copies, for mat_out is not contiguous in x axis
#pragma unroll
      for (uint32_t j = 0; j < max_head_size / mma_sg_tile_size; j++) {
        mat_sub_out.reg.xetla_select<mma_sg_tile_size, 1>(
            j * mma_sg_tile_size) =
            mat_out.reg.xetla_select<mma_sg_tile_size, 1>(
                j * mma_sg_tile_size * block_size_y + i * mma_sg_tile_size);
      }
      local_st_payload_t local_st_payload(
          slm_offset_out,
          ctx.sg_id_x * max_head_size + max_head_size,
          1,
          wg_size * max_head_size,
          ctx.sg_id_x * max_head_size,
          0);
      subgroup::tile_store(mat_sub_out, local_st_payload);

      xetla_fence<memory_kind::shared_local>();
      if constexpr (wg_size > 1)
        ctx.nbarrier.arrive_wait();

      // 2.1. loading slm out to reg
      using ld_tile_desc_t = subgroup::tile_desc_t<
          head_size_per_sg,
          wg_size,
          sg_tile_size_x,
          sg_tile_size_y>;
      using local_ld_tile_t = subgroup::tile_t<accum_t, ld_tile_desc_t>;
      using local_ld_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
          ld_tile_desc_t,
          msg_type::scatter,
          arch_tag>;

      local_ld_tile_t mat_out_ld;
      // load shared data [wg_size, head_size_per_sg]
      local_ld_payload_t local_ld_payload(
          slm_offset_out,
          start_x_local + head_size_per_sg,
          wg_size,
          max_head_size,
          start_x_local,
          0);
      subgroup::tile_load(mat_out_ld, local_ld_payload);

      //       // 2.2. do reduction along y dimension -> [head_size_per_sg]
      auto reduce_vec_acc =
          subgroup::tile_reduce<reduce_op::sum, accum_t, accum_t, 0>(
              mat_out_ld);
      auto reduce_vec =
          xetla_cvt<scalar_t, accum_t, head_size_per_sg>(reduce_vec_acc);
      // need multiple copies, for mat_out_st is not contiguous in x axis
#pragma unroll
      for (uint32_t j = 0; j < head_size_per_sg / sg_tile_size_x; j++) {
        mat_out_st.reg.xetla_select<sg_tile_size_x, 1>(
            i * sg_tile_size_x + j * sg_tile_size_x * block_size_y) =
            reduce_vec.xetla_select<sg_tile_size_x, 1>(sg_tile_size_x * j);
      }
    }
    // store out to global memory
    int32_t start_x = start_x_local;
    int32_t end_x = start_x + head_size_per_sg;
    end_x = end_x < max_head_size ? end_x : max_head_size;
    int32_t start_y = ctx.seq_q_start + ctx.group_id * block_size_y;
    int32_t end_y = start_y + block_size_y;
    end_y = end_y < ctx.seq_q_end ? end_y : ctx.seq_q_end;
    int32_t pitch = args.head_size * ctx.num_partitions * args.num_heads;
    auto out_ptr = args.out + ctx.partition_id * args.head_size +
        ctx.head_id * args.head_size * args.max_num_partitions;
    global_st_payload_t global_st_payload(
        out_ptr, end_x, end_y, pitch, start_x, start_y);
    subgroup::tile_store(mat_out_st, global_st_payload);
  }

 public:
  // Get the local memory size consumption.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_query + slm_size_softmax + slm_size_out;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  };

  // Get the named barrier consumption count.
  inline static constexpr uint32_t get_barrier_count() {
    return nbarrier_cnt;
  }

  // Helper function to get the nd_range.
  static sycl::nd_range<3> get_nd_range(
      uint32_t num_batches,
      uint32_t num_heads,
      uint32_t num_groups,
      uint32_t num_partitions) {
    sycl::range<3> local_range = sycl::range<3>{1, 1, wg_size};
    sycl::range<3> group_range =
        sycl::range<3>{num_heads * num_batches, num_groups, num_partitions};

    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  // Entrance of the functor
  inline KERNEL_FUNC void operator()(
      sycl::nd_item<3>& item,
      arguments_t& args) {
    // initialization
    ctx.init(item, args);

    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    preload_query(args);

    score_tile_t mat_score(-INFINITY);

    compute_score(mat_score, args);

    softmax(mat_score, args);
    out_tile_t mat_out(0.0f);

    compute_out(mat_score, mat_out, args);
    collect_out(mat_out, args);
  }
};

template <typename policy, typename scalar_t, gpu_arch arch_tag>
class chunked_prefill_reduce {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input and output tensors
    scalar_t* out; // [num_tokens, num_heads, head_size]
    scalar_t* tmp_out; // [num_tokens, num_heads, max_num_partitions, head_size]
    accum_t* max_logits; // [num_tokens, num_heads, max_num_partitions]
    accum_t* exp_sums; // [num_tokens, num_heads, max_num_partitions]
    // index_t* context_lens; // [num_seqs]

    int32_t* cu_seqlen_q;
    int32_t* cu_seqlen_k;
    uint32_t num_seqs;
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t max_num_partitions;

    inline arguments_t(
        scalar_t* out,
        scalar_t* tmp_out,
        accum_t* max_logits,
        accum_t* exp_sums,
        // index_t* context_lens,
        int32_t* cu_seqlen_q,
        int32_t* cu_seqlen_k,
        uint32_t num_seqs,
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t max_num_partitions)
        : out(out),
          tmp_out(tmp_out),
          max_logits(max_logits),
          exp_sums(exp_sums),
          // context_lens(context_lens),
          cu_seqlen_q(cu_seqlen_q),
          cu_seqlen_k(cu_seqlen_k),
          num_seqs(num_seqs),
          num_heads(num_heads),
          head_size(head_size),
          max_num_partitions(max_num_partitions) {}
  };

 private:
  // -------------------- // Compute policy // -------------------- //

  static constexpr accum_t neg_infinity = INFINITY * -1;
  static constexpr uint32_t stages = policy::stages;
  static constexpr uint32_t wg_size = policy::wg_size;
  static constexpr uint32_t block_m = policy::block_m;
  static constexpr uint32_t max_head_size = policy::max_head_size;
  static constexpr uint32_t head_size_stride = policy::head_size_stride;
  static constexpr uint32_t head_size_per_sg =
      std::max(max_head_size / wg_size, 16u);
  static constexpr uint32_t partition_size = policy::partition_size;
  static constexpr uint32_t partition_stride = policy::partition_stride;
  static constexpr uint32_t max_partitions_per_sg =
      policy::max_partitions_per_sg;

  // -------------------- // Slm and nbarrier // -------------------- //

  static constexpr uint32_t slm_size_reduce =
      (wg_size > 1) ? wg_size * block_m * sizeof(accum_t) : 0;
  static constexpr uint32_t slm_size_out =
      max_head_size * wg_size * sizeof(accum_t);

  static constexpr uint32_t slm_offset_reduce = 0;
  static constexpr uint32_t slm_offset_out =
      slm_offset_reduce + slm_size_reduce;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? 1 : 0;

  // -------------------- // Context // -------------------- //

  struct context_t {
    uint32_t sg_id;
    uint32_t seq_id;
    uint32_t head_id;
    uint32_t group_id;
    uint32_t context_len;

    uint32_t num_partitions;
    uint32_t wg_partition_stride;

    uint32_t num_partition_rows;

    xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;

    inline context_t() = default;

    inline void init(sycl::nd_item<3>& item, arguments_t& args) {
      sg_id = item.get_local_linear_id();
      seq_id = item.get_group(0) / args.num_heads;
      head_id = item.get_group(0) % args.num_heads;
      group_id = item.get_group(1);
      context_len = args.cu_seqlen_k[seq_id + 1] - args.cu_seqlen_k[seq_id];

      num_partitions = DIVIDE_ROUND_UP(context_len, partition_size);
      wg_partition_stride = wg_size * partition_stride;

      num_partition_rows = 0;

      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
    }

    inline void update_partition_num(uint32_t num) {
      num_partition_rows = num;
    }
  };

  context_t ctx;

  // -------------------- // load_data // -------------------- //

  using tile_desc_t = subgroup::tile_desc_t<
      partition_stride * max_partitions_per_sg,
      block_m,
      partition_stride,
      block_m>;
  using tile_t = subgroup::tile_t<accum_t, tile_desc_t>;

  inline void load_data(
      tile_t& src,
      accum_t* p,
      arguments_t& args,
      float init_value) {
    // TODO: using block_m as subgroup load size is a wa yet, will change to
    // more fine-graned size.
    using ld_tile_desc_t = subgroup::
        tile_desc_t<partition_stride, block_m, partition_stride, block_m>;
    using ld_tile_t = subgroup::tile_t<accum_t, ld_tile_desc_t>;
    using ld_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        ld_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    static constexpr tdesc_update_dir update_dir = tdesc_update_dir::x_dir;

    int32_t start_x = ctx.sg_id * partition_stride;
    int32_t start_y = args.cu_seqlen_q[ctx.seq_id] + ctx.group_id * block_m;
    int32_t end_y = start_y + block_m;
    int32_t boundary_y = args.cu_seqlen_q[ctx.seq_id + 1];
    end_y = end_y < boundary_y ? end_y : boundary_y;

    // offset the head id first, and load alone num_tokens as y,
    // max_num_partitions as x
    auto p_ptr = p + ctx.head_id * args.max_num_partitions;
    ld_tile_t ld_tile;
    ld_payload_t ld_payload(
        p_ptr,
        ctx.num_partitions,
        end_y,
        args.max_num_partitions * args.num_heads,
        start_x,
        start_y);

    for (int i = start_x, row_i = 0; i < ctx.num_partitions;
         i += ctx.wg_partition_stride, row_i++) {
      subgroup::tile_load(ld_tile, ld_payload);

      ld_payload.template update_tdesc<update_dir>(ctx.wg_partition_stride);
      int32_t remain = ctx.num_partitions - i;
#pragma unroll
      for (int block_idx = 0; block_idx < block_m; ++block_idx) {
        int32_t token_pos = start_y + block_idx;
        auto sub_reg = ld_tile.reg.xetla_select<partition_stride, 1>(
            block_idx * partition_stride);
        if (token_pos >= end_y) {
          sub_reg = init_value;
        } else {
          xetla_mask<partition_stride> mask =
              xetla_vector_gen<uint32_t, partition_stride>(1, 1) > remain;
          sub_reg.merge(init_value, mask);
        }
      }
      src.reg.xetla_select<partition_stride * block_m, 1>(
          row_i * partition_stride * block_m) = ld_tile.reg;

      ctx.update_partition_num(row_i + 1);
    }
  }

  // -------------------- // rescale_exp_sums // -------------------- //
  using reduce_tile_desc_t =
      subgroup::tile_desc_t<partition_stride * max_partitions_per_sg, 1, 1, 1>;
  using reduce_tile_t = subgroup::tile_t<accum_t, reduce_tile_desc_t>;

  inline void rescale_exp_sums(tile_t& mat_exp_sums, arguments_t& args) {
    tile_t mat_max_logits(neg_infinity);
    load_data(mat_max_logits, args.max_logits, args, neg_infinity);
    using wg_row_max_t =
        fmha::group_row_reduce_t<tile_t, wg_size, reduce_op::max, arch_tag>;
    wg_row_max_t wg_row_max(ctx.sg_id, 0, slm_offset_reduce);
    xetla_vector<accum_t, block_m> max_logits = wg_row_max(mat_max_logits);

    if constexpr (wg_size > 1) {
      ctx.nbarrier.arrive();
    }

    load_data(mat_exp_sums, args.exp_sums, args, 0);

    subgroup::tile_broadcast_op<subgroup::tile_minus, tile_t>(
        mat_max_logits, max_logits);

    mat_exp_sums.reg =
        mat_exp_sums.reg * xetla_exp<accum_t>(mat_max_logits.reg);

    if constexpr (wg_size > 1) {
      ctx.nbarrier.wait();
    }
  }

  // -------------------- // compute_out // -------------------- //

  using out_tile_desc_t =
      subgroup::tile_desc_t<max_head_size, block_m, head_size_stride, 1>;
  using out_tile_t = subgroup::tile_t<accum_t, out_tile_desc_t>;

  inline void compute_out(
      tile_t& rescaled_exp_sums,
      out_tile_t& mat_out,
      arguments_t& args) {
    constexpr uint32_t sg_tile_size_x = partition_stride > 32 / sizeof(scalar_t)
        ? 32 / sizeof(scalar_t)
        : partition_stride;
    constexpr uint32_t sg_tile_size_y =
        head_size_stride > 16 ? 16 : head_size_stride;

    // for loading tmp out to register
    using tmp_tile_desc_t = subgroup::tile_desc_t<
        partition_stride,
        head_size_stride,
        sg_tile_size_x,
        sg_tile_size_y>;
    using tmp_tile_t = subgroup::tile_t<scalar_t, tmp_tile_desc_t>;
    using tmp_acc_tile_t = subgroup::tile_t<accum_t, tmp_tile_desc_t>;
    using tmp_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>,
        tmp_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    using tmp_prefetch_payload_t = subgroup::prefetch_payload_t<
        mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>,
        tmp_tile_desc_t,
        1,
        arch_tag>;
    constexpr tdesc_update_dir tmp_update_dir = tdesc_update_dir::x_dir;

    using wg_row_sum_t =
        fmha::group_row_reduce_t<tile_t, wg_size, reduce_op::sum, arch_tag>;
    wg_row_sum_t wg_row_sum(ctx.sg_id, 0, slm_offset_reduce);
    xetla_vector<accum_t, block_m> global_exp_sum =
        wg_row_sum(rescaled_exp_sums);

    auto tmp_out_ptr = args.tmp_out;
#pragma unroll
    for (int i = 0; i < block_m; ++i) {
      accum_t inv_global_exp_sum_sub = 1.0f / global_exp_sum[i];
      tmp_tile_t mat_tmp_out(0);
      const int loop_count = DIVIDE_ROUND_UP(args.head_size, head_size_stride);
      int32_t token_pos =
          args.cu_seqlen_q[ctx.seq_id] + ctx.group_id * block_m + i;
      if (token_pos >= args.cu_seqlen_q[ctx.seq_id + 1])
        break;
      int32_t base_x = (token_pos * args.num_heads + ctx.head_id) *
          args.max_num_partitions * args.head_size;
      tmp_out_ptr = args.tmp_out +
          (token_pos * args.num_heads + ctx.head_id) * args.max_num_partitions *
              args.head_size;

      for (int start_x = ctx.sg_id * partition_stride, row_i = 0;
           start_x < ctx.num_partitions;
           start_x += ctx.wg_partition_stride, row_i++) {
        tmp_payload_t tmp_payload(
            tmp_out_ptr,
            args.head_size,
            ctx.num_partitions,
            args.head_size,
            0,
            start_x);
        tmp_prefetch_payload_t tmp_prefetch_payload(
            tmp_out_ptr,
            args.head_size,
            ctx.num_partitions,
            args.head_size,
            0,
            start_x);
#if 0
#pragma unroll
        for (int stage = 0; i < stages; ++stage) {
          subgroup::tile_prefetch(tmp_prefetch_payload);
          tmp_prefetch_payload.template update_tdesc<tmp_update_dir>(
              head_size_stride);
        }
#endif

        auto rescaled_exp_sums_sub_slice =
            rescaled_exp_sums.reg.xetla_select<partition_stride, 1>(
                row_i * partition_stride * block_m + i * partition_stride);
        rescaled_exp_sums_sub_slice =
            rescaled_exp_sums_sub_slice * inv_global_exp_sum_sub;

        for (int j = 0; j < loop_count; j++) {
          subgroup::tile_load(mat_tmp_out, tmp_payload);

#if 0
          if constexpr (stages != 0) {
            subgroup::tile_prefetch(tmp_prefetch_payload);
          }
#endif
          //   SW_BARRIER();
          tmp_payload.template update_tdesc<tmp_update_dir>(head_size_stride);
#if 0
          if constexpr (stages != 0) {
            tmp_prefetch_payload.template update_tdesc<tmp_update_dir>(
                head_size_stride);
          }
#endif
          //   SW_BARRIER();
          tmp_acc_tile_t mat_tmp_out_acc;
          subgroup::elemwise_cvt(mat_tmp_out_acc, mat_tmp_out);
          //   SW_BARRIER();
          auto out_sub = mat_out.reg.xetla_select<head_size_stride, 1>(
              i * max_head_size + j * head_size_stride);
          out_sub += mat_vec_mul<accum_t, partition_stride, tmp_acc_tile_t, 1>(
              rescaled_exp_sums_sub_slice, mat_tmp_out_acc);

          //   SW_BARRIER();
        }
        // #endif
      }
    }
  }

  // -------------------- // compute_out // -------------------- //
  using out_tile_slice_desc_t = subgroup::tile_desc_t<max_head_size, 1, 16, 1>;
  using out_tile_slice_t = subgroup::tile_t<accum_t, out_tile_slice_desc_t>;
  inline void collect_out(out_tile_t& mat_out, arguments_t& args) {
    constexpr uint32_t sg_tile_size_x = head_size_per_sg > 32 / sizeof(scalar_t)
        ? 32 / sizeof(scalar_t)
        : head_size_per_sg;
    constexpr uint32_t sg_tile_size_y = wg_size > 16 ? 16 : wg_size;

    // for storing out to slm
    using local_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        out_tile_slice_desc_t,
        msg_type::scatter,
        arch_tag>;
    // for loading out to reg
    using ld_tile_desc_t = subgroup::
        tile_desc_t<head_size_per_sg, wg_size, sg_tile_size_x, sg_tile_size_y>;
    using local_ld_tile_t = subgroup::tile_t<accum_t, ld_tile_desc_t>;
    using local_ld_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        ld_tile_desc_t,
        msg_type::scatter,
        arch_tag>;
    // for storing out to global
    using st_tile_desc_t = subgroup::
        tile_desc_t<head_size_per_sg, block_m, sg_tile_size_x, block_m>;
    using global_st_tile_t = subgroup::tile_t<scalar_t, st_tile_desc_t>;
    using global_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        st_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    // store out data of each subgroup to slm
    // int32_t start_y = ctx.sg_id;
    out_tile_slice_t mat_out_slice(0.0f);
    global_st_tile_t global_st_tile(0.0f);

    uint32_t boundary_x = args.head_size;
    uint32_t boundary_y = args.cu_seqlen_q[ctx.seq_id + 1];
    uint32_t start_x = ctx.sg_id * head_size_per_sg;
    uint32_t start_y = args.cu_seqlen_q[ctx.seq_id] + ctx.group_id * block_m;
    uint32_t pitch = args.head_size * args.num_heads;

#pragma unroll
    for (int i = 0; i < block_m; ++i) {
      uint32_t token_offset = start_y + i;
      if (token_offset >= boundary_y)
        break;
      mat_out_slice.reg =
          mat_out.reg.xetla_select<max_head_size, 1>(i * max_head_size);
      local_st_payload_t local_st_payload(
          slm_offset_out, max_head_size, wg_size, max_head_size, 0, ctx.sg_id);

      subgroup::tile_store(mat_out_slice, local_st_payload);

      xetla_fence<memory_kind::shared_local>();
      if constexpr (wg_size > 1)
        ctx.nbarrier.arrive_wait();
      if (start_x < args.head_size) {
        local_ld_tile_t mat_out_ld;
        local_ld_payload_t local_ld_payload(
            slm_offset_out, max_head_size, wg_size, max_head_size, start_x, 0);
        subgroup::tile_load(mat_out_ld, local_ld_payload);

        auto global_st_tile_slice =
            global_st_tile.reg.xetla_select<head_size_per_sg, 1>(
                i * head_size_per_sg);
        global_st_tile_slice = xetla_cvt<scalar_t, accum_t, head_size_per_sg>(
            subgroup::tile_reduce<reduce_op::sum, accum_t, accum_t, 0>(
                mat_out_ld));
      }
    }

    auto out_ptr = args.out + ctx.head_id * args.head_size;
    global_st_payload_t global_st_payload(
        out_ptr, boundary_x, boundary_y, pitch, start_x, start_y);
    subgroup::tile_store(global_st_tile, global_st_payload);
  }

 public:
  // Get the local memory size consumption.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_reduce + slm_size_out;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  };

  // Get the named barrier consumption count.
  inline static constexpr uint32_t get_barrier_count() {
    return nbarrier_cnt;
  }

  // Helper function to get the nd_range.
  static sycl::nd_range<3> get_nd_range(
      uint32_t num_seqs,
      uint32_t num_heads,
      uint32_t num_groups) {
    sycl::range<3> local_range = sycl::range<3>{1, 1, wg_size};
    sycl::range<3> group_range =
        sycl::range<3>{num_seqs * num_heads, num_groups, 1};

    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  // Entrance of the functor
  inline KERNEL_FUNC void operator()(
      sycl::nd_item<3>& item,
      arguments_t& args) {
    // initialization
    ctx.init(item, args);
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    tile_t mat_exp_sums(0.0f);
    rescale_exp_sums(mat_exp_sums, args);
    out_tile_t mat_out(0.0f);

    compute_out(mat_exp_sums, mat_out, args);

    collect_out(mat_out, args);
    // #endif
  }
};

#undef DIVIDE_ROUND_UP

} // namespace attention

} // namespace gpu::xetla
