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
#include <cstdint>
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

template <
    typename policy,
    typename scalar_t,
    typename kv_t,
    typename index_t,
    gpu_arch arch_tag,
    fp8_format fp8_format_ = fp8_format::E4M3>
class paged_attention_kernel_vllm {
 public:
  using accum_t = float;

  struct unused_t {};

  struct arguments_t {
    // Input and output tensors
    accum_t* max_logits; // [num_seqs, num_heads, max_num_partitions]
    accum_t* exp_sums; // [num_seqs, num_heads, max_num_partitions]
    scalar_t* out; // [num_seqs, num_heads, max_num_partitions, head_size]
    scalar_t* query; // [num_seqs, num_heads, head_size]
    kv_t* key_cache; // [num_blocks, block_size, num_kv_heads, head_size]
    kv_t* value_cache; // [num_blocks, block_size, num_kv_heads, head_size]
    scalar_t* sinks; // [num_heads]
    float* alibi_slopes; // [num_heads] - alibi_slopes

    // Index
    index_t* block_tables; // [num_seqs, max_blocks_per_seq]
    index_t* context_lens; // [num_seqs]

    // Softmax scale
    accum_t sm_scale;

    // Dimension size
    uint32_t num_seqs;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_size;
    uint32_t max_blocks_per_seq;
    accum_t softcap;
    float fp8_scale;

    inline arguments_t(
        accum_t* max_logits,
        accum_t* exp_sums,
        scalar_t* out,
        scalar_t* query,
        kv_t* key_cache,
        kv_t* value_cache,
        scalar_t* sinks,
        float* alibi_slopes,
        index_t* block_tables,
        index_t* context_lens,
        accum_t sm_scale,
        uint32_t num_seqs,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t max_blocks_per_seq,
        accum_t softcap,
        float fp8_scale = 1.0f)
        : max_logits(max_logits),
          exp_sums(exp_sums),
          out(out),
          query(query),
          key_cache(key_cache),
          value_cache(value_cache),
          sinks(sinks),
          alibi_slopes(alibi_slopes),
          block_tables(block_tables),
          context_lens(context_lens),
          sm_scale(sm_scale),
          num_seqs(num_seqs),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_size(head_size),
          max_blocks_per_seq(max_blocks_per_seq),
          softcap(softcap),
          fp8_scale(fp8_scale) {}
  };

  using tanh_t = typename subgroup::tanh_op_t;

 private:
  // -------------------- // Compute policy // -------------------- //

  static constexpr accum_t neg_infinity = INFINITY * -1;
  static constexpr uint32_t stages = policy::stages;
  static constexpr uint32_t wg_size = policy::wg_size;
  static constexpr uint32_t block_size = policy::block_size;
  static constexpr uint32_t max_head_size = policy::max_head_size;
  static constexpr uint32_t head_size_stride = policy::head_size_stride;
  static constexpr uint32_t max_blocks_per_sg = policy::max_blocks_per_sg;
  static constexpr uint32_t query_group_size = policy::query_group_size;

  // used for preload query and store output
  // use minimum 16 to avoid mask in load
  static constexpr uint32_t head_size_per_sg =
      std::max(max_head_size / wg_size, 16u);
  static constexpr uint32_t partition_size = policy::partition_size;
  static constexpr bool use_partition = partition_size > 0;
  static constexpr uint32_t value_blocks_per_sg = partition_size / block_size;

  // -------------------- // Slm and nbarrier // -------------------- //

  static constexpr uint32_t slm_size_score =
      query_group_size * partition_size * sizeof(accum_t);
  static constexpr uint32_t slm_size_exp_score =
      query_group_size * partition_size * sizeof(scalar_t);

  static constexpr uint32_t slm_offset_score = 0;
  static constexpr uint32_t slm_offset_exp_score =
      slm_offset_score + slm_size_score;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? 1 : 0;
  // This boolean variable will determine whether the kernel will execute 2d
  // load path, the original 2d load paged attention path is disabled by default
  // here to prevent the accuracy issue brought by the surface width limitation
  // of 2d load.
  static constexpr bool has_2d_ld_st =
      (arch_tag == gpu_arch::XeHpc) ? true : false;

  static constexpr int loop_count =
      DIVIDE_ROUND_UP(max_head_size, head_size_stride);

  // -------------------- // Context // -------------------- //

  struct context_t {
    uint32_t sg_id;
    uint32_t seq_id;
    uint32_t head_id;
    uint32_t partition_id;
    uint32_t max_num_partitions;
    uint32_t context_len;
    int kv_head_id;

    index_t* block_table; // [max_blocks_per_seq]
    uint32_t num_blocks_per_sg; // number of blocks processed by current sg

    int kv_block_stride;
    int kv_head_stride;
    int start_block_id;
    int end_block_id;

    float alibi_slopes;
    accum_t sinks_array[query_group_size];

    xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;

    inline context_t() = default;

    inline void init(sycl::nd_item<3>& item, arguments_t& args) {
      sg_id = item.get_local_linear_id();
      kv_head_id = item.get_group(1);
      seq_id = item.get_group(0);
      partition_id = item.get_group(2);
      max_num_partitions = item.get_group_range(2);

      // head_id = kv_head_id * query_group_size;
      // if (args.alibi_slopes != nullptr) {
      //   alibi_slopes = args.alibi_slopes[head_id];
      // }
      if (args.sinks != nullptr) {
#pragma unroll
        for (uint32_t i = 0; i < query_group_size; i++) {
          sinks_array[i] = static_cast<accum_t>(
              args.sinks[kv_head_id * query_group_size + i]);
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < query_group_size; i++) {
          sinks_array[i] = neg_infinity;
        }
      }

      context_len = args.context_lens[seq_id];
      block_table = args.block_tables + seq_id * args.max_blocks_per_seq;
      num_blocks_per_sg = 0;

      kv_block_stride = args.num_kv_heads * max_head_size * block_size;
      kv_head_stride = max_head_size * kv_head_id;

      const int max_num_blocks = DIVIDE_ROUND_UP(context_len, block_size);
      const int num_blocks_per_wg =
          use_partition ? partition_size / block_size : max_num_blocks;

      start_block_id = partition_id * num_blocks_per_wg;
      end_block_id =
          std::min(max_num_blocks, start_block_id + num_blocks_per_wg);

      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
    }

    inline void update_block_num(uint32_t num) {
      num_blocks_per_sg = num;
    }
  };

  context_t ctx;

  // -------------------- // compute_score // -------------------- //

  static constexpr uint32_t mma_sg_tile_size = 16;
  using score_tile_desc_t = subgroup::tile_desc_t<
      block_size * max_blocks_per_sg,
      query_group_size,
      mma_sg_tile_size,
      query_group_size>;
  using score_tile_t = subgroup::tile_t<accum_t, score_tile_desc_t>;

  // Compute query x key.
  inline void compute_score(arguments_t& args) {
    constexpr uint32_t sg_tile_size_head =
        head_size_stride > 32 / sizeof(scalar_t) ? 32 / sizeof(scalar_t)
                                                 : head_size_stride; // 16
    constexpr uint32_t sg_tile_size_block = mma_sg_tile_size;

    using score_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        score_tile_desc_t,
        msg_type::scatter,
        arch_tag>;

    using query_tile_desc_t = subgroup::tile_desc_t<
        head_size_stride,
        query_group_size,
        sg_tile_size_head,
        query_group_size>;
    using query_tile_t = subgroup::tile_t<scalar_t, query_tile_desc_t>;
    using query_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        query_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    constexpr tdesc_update_dir query_update_dir = tdesc_update_dir::x_dir;
    using key_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        head_size_stride,
        sg_tile_size_block,
        sg_tile_size_head,
        reg_layout::vnni_tiled>;
    constexpr mem_layout k_mem_layout = mem_layout::col_major;
    constexpr tdesc_update_dir key_update_dir = tdesc_update_dir::x_dir;

    using key_mma_tile_t = subgroup::tile_t<scalar_t, key_tile_desc_t>;
    using key_tile_t = subgroup::tile_t<kv_t, key_tile_desc_t>;
    using key_payload_t = subgroup::mem_payload_t<
        mem_desc_t<kv_t, k_mem_layout, mem_space::global>,
        key_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    using score_acc_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        query_group_size,
        mma_sg_tile_size,
        query_group_size>;
    using score_acc_tile_t = subgroup::tile_t<accum_t, score_acc_tile_desc_t>;

    using tile_mma = subgroup::tile_mma_t<
        score_acc_tile_t,
        score_acc_tile_t,
        key_mma_tile_t,
        query_tile_t,
        mma_engine::xmx,
        arch_tag>;

    using dequantize_key_t = std::conditional_t<
        std::is_same_v<kv_t, uint8_t>,
        subgroup::dequant_fp8_weight_t<
            key_mma_tile_t,
            key_tile_t,
            fp8_format_,
            false>,
        unused_t>;

    query_tile_t mat_query;
    key_tile_t mat_key;
    key_mma_tile_t mat_key_mma;
    score_acc_tile_t score_sub(neg_infinity);
    dequantize_key_t dequant_key;

    constexpr uint32_t boundary_score_y = query_group_size;
    uint32_t start_score_x = ctx.sg_id * block_size;
    uint32_t boundary_score_x = start_score_x + block_size;

    score_payload_t score_payload(
        slm_offset_score,
        boundary_score_x,
        boundary_score_y,
        partition_size,
        start_score_x,
        0);

    // iterate over context blocks
    for (int bid = ctx.sg_id + ctx.start_block_id, row_i = 0;
         bid < ctx.end_block_id;
         bid += wg_size, row_i++) {
      ctx.update_block_num(row_i + 1);

      // get the physical block id from block_table
      // Note, we didn't add correct boundary for context length, as we handled
      // this in following mask
      const int block_id = ctx.block_table[bid];

      int32_t start_y = block_id * block_size;
      uint32_t boundary_y = start_y + block_size;
      int32_t start_x = ctx.kv_head_stride;
      uint32_t boundary_x = start_x + max_head_size;
      uint32_t pitch = max_head_size * args.num_kv_heads;
      auto* cur_key_cache = args.key_cache;
      key_payload_t key_payload(
          cur_key_cache, boundary_x, boundary_y, pitch, start_x, start_y);

      constexpr uint32_t boundary_query_y = query_group_size;
      constexpr uint32_t boundary_query_x = max_head_size;
      auto* cur_query = args.query +
          ctx.seq_id * args.num_heads * max_head_size +
          ctx.kv_head_id * query_group_size * max_head_size;

      query_payload_t query_payload(
          cur_query, boundary_query_x, boundary_query_y, max_head_size, 0, 0);

      score_sub.reg = 0;
#pragma unroll
      for (int i = 0; i < loop_count; i++) {
        subgroup::tile_load(mat_query, query_payload);
        subgroup::tile_load(mat_key, key_payload);

        query_payload.template update_tdesc<query_update_dir>(head_size_stride);
        key_payload.template update_tdesc<key_update_dir>(head_size_stride);

        if constexpr (std::is_same_v<kv_t, uint8_t>) {
          dequant_key(mat_key_mma, mat_key, args.fp8_scale);
        } else {
          mat_key_mma = mat_key;
        }

        SW_BARRIER();
        tile_mma::mma(score_sub, score_sub, mat_key_mma, mat_query);
        SW_BARRIER();
      }
      score_sub.reg *= args.sm_scale;

      if constexpr (std::is_same_v<kv_t, uint8_t>) {
        score_sub.reg *= args.fp8_scale;
      }

      uint32_t remained_len = ctx.context_len - bid * block_size;
      using remain_tile_mask = fmha::tile_mask_t<score_acc_tile_t>;
      if (remained_len < block_size) {
        remain_tile_mask::padding_mask(score_sub, remained_len);
      }

      // if (args.softcap > 0.0) {
      //   score_sub.reg /= args.softcap;
      //   tanh_t tanh;
      //   tanh(score_sub, 0);
      //   score_sub.reg *= args.softcap;
      // }

      // if (args.alibi_slopes != nullptr) {
      //   int32_t mat_real_x = bid * block_size;
      //   int32_t mat_real_y = ctx.seq_id;
      //   xetla_vector<float, block_size> pos_id =
      //       xetla_vector_gen<float, block_size>(mat_real_x, 1);
      //   score_sub += (pos_id * ctx.alibi_slopes);
      // }
    }
    subgroup::tile_store(score_sub, score_payload);
    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

  // -------------------- // softmax // -------------------- //

  // Compute softmax of score.
  inline void softmax(arguments_t& args) {
    for (int row_id = ctx.sg_id; row_id < query_group_size; row_id += wg_size) {
      using softmax_score_tile_desc_t =
          subgroup::tile_desc_t<partition_size, 1, mma_sg_tile_size, 1>;
      using softmax_score_tile_t =
          subgroup::tile_t<accum_t, softmax_score_tile_desc_t>;
      softmax_score_tile_t sm_mat_score;

      using sm_local_ld_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
          softmax_score_tile_desc_t,
          msg_type::block_1d,
          arch_tag>;
      using sm_local_st_payload_t = subgroup::mem_payload_t<
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
          softmax_score_tile_desc_t,
          msg_type::block_1d,
          arch_tag>;

      uint32_t start_y = row_id;
      constexpr uint32_t boundary_y = query_group_size;
      sm_local_ld_payload_t sm_local_ld_payload(
          slm_offset_score,
          partition_size,
          boundary_y,
          partition_size,
          0,
          start_y);

      subgroup::tile_load(sm_mat_score, sm_local_ld_payload);
      xetla_vector<accum_t, 1> max_score =
          subgroup::tile_reduce<reduce_op::max, accum_t, accum_t, 1>(
              sm_mat_score);

      accum_t sink =
          ctx.partition_id == 0 ? ctx.sinks_array[row_id] : neg_infinity;
      accum_t scalar_max = max_score[0];
      scalar_max = std::max(scalar_max, sink);
      sm_mat_score.reg -= scalar_max;
      sm_mat_score.reg = xetla_exp<accum_t>(sm_mat_score.reg);
      sink = std::exp(sink - scalar_max);

      xetla_vector<accum_t, 1> sum_score =
          subgroup::tile_reduce<reduce_op::sum, accum_t, accum_t, 1>(
              sm_mat_score);
      accum_t scalar_sum = sum_score[0];
      scalar_sum += sink;
      sm_mat_score.reg /= scalar_sum;

      // Store the softmax result back to shared local memory
      using softmax_exp_score_tile_t =
          subgroup::tile_t<scalar_t, softmax_score_tile_desc_t>;
      softmax_exp_score_tile_t sm_mat_exp_score;
      subgroup::elemwise_cvt(sm_mat_exp_score, sm_mat_score);

      sm_local_st_payload_t sm_local_st_payload(
          slm_offset_exp_score,
          partition_size,
          boundary_y,
          partition_size,
          0,
          start_y);
      subgroup::tile_store(sm_mat_exp_score, sm_local_st_payload);

      // Store max and sum
      using scalar_tile_desc_t = subgroup::tile_desc_t<1, 1, 1, 1>;
      using scalar_tile_t = subgroup::tile_t<accum_t, scalar_tile_desc_t>;
      using global_scalar_st_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
          scalar_tile_desc_t,
          msg_type::block_1d,
          arch_tag>;

      uint32_t start_g_x = ctx.partition_id;
      uint32_t boundary_g_x = ctx.max_num_partitions;
      uint32_t start_g_y = row_id;
      constexpr uint32_t boundary_g_y = query_group_size;

      auto* cur_max_logits = args.max_logits +
          ctx.seq_id * args.num_heads * ctx.max_num_partitions +
          ctx.kv_head_id * query_group_size * ctx.max_num_partitions;
      global_scalar_st_payload_t max_logits_st_payload(
          cur_max_logits,
          boundary_g_x,
          boundary_g_y,
          ctx.max_num_partitions,
          start_g_x,
          start_g_y);
      scalar_tile_t max_logit_scalar(scalar_max);
      subgroup::tile_store(max_logit_scalar, max_logits_st_payload);

      auto* cur_exp_sums = args.exp_sums +
          ctx.seq_id * args.num_heads * ctx.max_num_partitions +
          ctx.kv_head_id * query_group_size * ctx.max_num_partitions;
      global_scalar_st_payload_t exp_sums_st_payload(
          cur_exp_sums,
          boundary_g_x,
          boundary_g_y,
          ctx.max_num_partitions,
          start_g_x,
          start_g_y);
      scalar_tile_t exp_sum_scalar(scalar_sum);
      subgroup::tile_store(exp_sum_scalar, exp_sums_st_payload);
    }
    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

  // -------------------- // compute_out // -------------------- //

  // Compute output.
  inline void compute_out(arguments_t& args) {
    constexpr uint32_t sg_tile_size_head = std::min(
        uint32_t(head_size_stride), uint32_t(32 / sizeof(scalar_t))); // 16
    constexpr uint32_t sg_tile_size_block =
        std::min(uint32_t(block_size), 32u); // 32
    constexpr uint32_t sg_tile_size_x =
        std::min(uint32_t(block_size), uint32_t(32 / sizeof(scalar_t))); // 16
    constexpr uint32_t sg_tile_size_y = 16;

    using value_tile_desc_t = subgroup::tile_desc_t<
        head_size_per_sg,
        block_size,
        sg_tile_size_x,
        sg_tile_size_y,
        reg_layout::vnni_tiled>;
    using value_tile_t = subgroup::tile_t<kv_t, value_tile_desc_t>;
    using value_mma_tile_t = subgroup::tile_t<scalar_t, value_tile_desc_t>;
    using value_payload_t = subgroup::mem_payload_t<
        mem_desc_t<kv_t, mem_layout::row_major, mem_space::global>,
        value_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    using exp_score_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        query_group_size,
        sg_tile_size_head,
        query_group_size>;
    using exp_score_tile_t = subgroup::tile_t<scalar_t, exp_score_tile_desc_t>;
    using exp_score_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
        exp_score_tile_desc_t,
        msg_type::scatter,
        arch_tag>;
    constexpr tdesc_update_dir exp_score_update_dir = tdesc_update_dir::x_dir;

    constexpr uint32_t boundary_e_y = query_group_size;
    constexpr uint32_t boundary_e_x = block_size;
    constexpr uint32_t pitch_e = partition_size;
    exp_score_payload_t exp_score_payload(
        slm_offset_exp_score, boundary_e_x, boundary_e_y, pitch_e, 0, 0);

    using out_tile_desc_t = subgroup::tile_desc_t<
        head_size_per_sg,
        query_group_size,
        head_size_per_sg,
        query_group_size>;
    using out_acc_tile_t = subgroup::tile_t<accum_t, out_tile_desc_t>;
    using out_tile_t = subgroup::tile_t<scalar_t, out_tile_desc_t>;

    using tile_mma = subgroup::tile_mma_t<
        out_acc_tile_t,
        out_acc_tile_t,
        value_mma_tile_t,
        exp_score_tile_t,
        mma_engine::xmx,
        arch_tag>;

    using dequant_value_t = std::conditional_t<
        std::is_same_v<kv_t, uint8_t>,
        subgroup::dequant_fp8_weight_t<
            value_mma_tile_t,
            value_tile_t,
            fp8_format_,
            false>,
        unused_t>;

    exp_score_tile_t mat_exp_score;
    value_tile_t mat_value;
    value_mma_tile_t mat_value_mma;
    out_acc_tile_t mat_acc_out(0.0f);
    dequant_value_t dequant_value;

#pragma unroll
    for (uint32_t i = 0; i < value_blocks_per_sg; ++i) {
      uint32_t cur_bid = i + ctx.start_block_id;
      if (cur_bid >= ctx.end_block_id)
        break;

      uint32_t block_id = ctx.block_table[cur_bid];

      constexpr uint32_t boundary_v_y = block_size;
      uint32_t start_v_x =
          ctx.kv_head_id * max_head_size + ctx.sg_id * head_size_per_sg;
      uint32_t boundary_v_x = start_v_x + head_size_per_sg;
      uint32_t pitch_v = args.num_kv_heads * max_head_size;
      auto* cur_value_cache = args.value_cache + block_id * ctx.kv_block_stride;
      value_payload_t value_payload(
          cur_value_cache, boundary_v_x, boundary_v_y, pitch_v, start_v_x, 0);

      subgroup::tile_load(mat_exp_score, exp_score_payload);
      subgroup::tile_load(mat_value, value_payload);

      exp_score_payload.template update_tdesc<exp_score_update_dir>(block_size);

      if constexpr (std::is_same_v<kv_t, uint8_t>) {
        dequant_value(mat_value_mma, mat_value, args.fp8_scale);
      } else {
        mat_value_mma = mat_value;
      }

      SW_BARRIER();
      tile_mma::mma(mat_acc_out, mat_acc_out, mat_value_mma, mat_exp_score);
      SW_BARRIER();
    }

    if constexpr (std::is_same_v<kv_t, uint8_t>) {
      mat_acc_out.reg *= args.fp8_scale;
    }

    out_tile_t mat_out;
    subgroup::elemwise_cvt(mat_out, mat_acc_out);

    using out_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        out_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    uint32_t start_o_x =
        ctx.partition_id * max_head_size + ctx.sg_id * head_size_per_sg;
    uint32_t boundary_o_x = start_o_x + head_size_per_sg;
    constexpr uint32_t boundary_o_y = query_group_size;
    uint32_t pitch_o = ctx.max_num_partitions * max_head_size;
    auto* cur_out = args.out +
        ctx.seq_id * args.num_heads * ctx.max_num_partitions * max_head_size +
        ctx.kv_head_id * query_group_size * ctx.max_num_partitions *
            max_head_size;

    out_st_payload_t out_st_payload(
        cur_out, boundary_o_x, boundary_o_y, pitch_o, start_o_x, 0);

    subgroup::tile_store(mat_out, out_st_payload);
  }

 public:
  // Get the local memory size consumption.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_score + slm_size_exp_score;
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
  static inline sycl::nd_range<3> get_nd_range(
      uint32_t num_seqs,
      uint32_t num_kv_heads,
      uint32_t max_num_partitions) {
    static const sycl::range<3> local_range = sycl::range<3>{1, 1, wg_size};
    sycl::range<3> group_range =
        sycl::range<3>{num_seqs, num_kv_heads, max_num_partitions};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  // Entrance of the functor
  inline SYCL_ESIMD_FUNCTION void operator()(
      sycl::nd_item<3>& item,
      arguments_t& args) {
    // initialization

    ctx.init(item, args);
    if (use_partition && ctx.start_block_id >= ctx.end_block_id) {
      return;
    }

    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    compute_score(args);
    softmax(args);
    compute_out(args);
  }
};

// ======================== // Reduce kernel // ======================== //

template <
    typename policy,
    typename scalar_t,
    typename kv_t,
    typename index_t,
    gpu_arch arch_tag,
    fp8_format fp8_format_>
class paged_attention_reduce_vllm {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input and output tensors
    scalar_t* out; // [num_seqs, num_heads, head_size]
    scalar_t* tmp_out; // [num_seqs, num_heads, max_num_partitions, head_size]
    accum_t* max_logits; // [num_seqs, num_heads, max_num_partitions]
    accum_t* exp_sums; // [num_seqs, num_heads, max_num_partitions]
    index_t* context_lens; // [num_seqs]

    uint32_t num_seqs;
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t max_num_partitions;

    inline arguments_t(
        scalar_t* out,
        scalar_t* tmp_out,
        accum_t* max_logits,
        accum_t* exp_sums,
        index_t* context_lens,
        uint32_t num_seqs,
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t max_num_partitions)
        : out(out),
          tmp_out(tmp_out),
          max_logits(max_logits),
          exp_sums(exp_sums),
          context_lens(context_lens),
          num_seqs(num_seqs),
          num_heads(num_heads),
          head_size(head_size),
          max_num_partitions(max_num_partitions) {}
  };
  using tanh_t = typename subgroup::tanh_op_t;

 private:
  // -------------------- // Compute policy // -------------------- //

  static constexpr accum_t neg_infinity = INFINITY * -1;
  static constexpr uint32_t stages = policy::stages;
  static constexpr uint32_t wg_size = policy::wg_size;
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
      (wg_size > 1) ? wg_size * sizeof(accum_t) : 0;
  static constexpr uint32_t slm_size_out =
      max_head_size * wg_size * sizeof(accum_t);

  static constexpr uint32_t slm_offset_reduce = 0;
  static constexpr uint32_t slm_offset_out =
      slm_offset_reduce + slm_size_reduce;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? 1 : 0;
  // This boolean variable will determine whether the kernel will execute 2d
  // load path, Ideally, the 2d load path always be the default choice for
  // machine which support 2d load instruction for better performance.
  static constexpr bool has_2d_ld_st =
      gpu::xetla::arch_has_2d_load_store<arch_tag>;
  ;

  // -------------------- // Context // -------------------- //

  struct context_t {
    uint32_t sg_id;
    uint32_t seq_id;
    uint32_t head_id;
    uint32_t context_len;

    uint32_t num_partitions;
    uint32_t wg_partition_stride;

    uint32_t num_partition_rows;

    xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;

    inline context_t() = default;

    inline void init(sycl::nd_item<3>& item, arguments_t& args) {
      sg_id = item.get_local_linear_id();
      seq_id = item.get_group(0);
      head_id = item.get_group(1);
      context_len = args.context_lens[seq_id];

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
      partition_stride,
      max_partitions_per_sg,
      partition_stride,
      max_partitions_per_sg>;
  using tile_t = subgroup::tile_t<accum_t, tile_desc_t>;

  inline void load_data(
      tile_t& src,
      accum_t* p,
      arguments_t& args,
      float init_value) {
    using ld_tile_desc_t = subgroup::tile_desc_t<partition_stride, 1, 1, 1>;
    using ld_tile_t = subgroup::tile_t<accum_t, ld_tile_desc_t>;
    using ld_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        ld_tile_desc_t,
        msg_type::block_1d,
        arch_tag>;
    static constexpr tdesc_update_dir update_dir = tdesc_update_dir::x_dir;

    int32_t start_x = ctx.sg_id * partition_stride;
    int32_t start_y = ctx.seq_id * args.num_heads + ctx.head_id;
    uint32_t boundary_y = start_y + 1;

    ld_tile_t ld_tile;
    ld_payload_t ld_payload(
        p,
        ctx.num_partitions,
        boundary_y,
        args.max_num_partitions,
        start_x,
        start_y);

    for (int i = start_x, row_i = 0; i < ctx.num_partitions;
         i += ctx.wg_partition_stride, row_i++) {
      subgroup::tile_load(ld_tile, ld_payload);
      ld_payload.template update_tdesc<update_dir>(ctx.wg_partition_stride);
      auto src_sub =
          src.reg.xetla_select<partition_stride, 1>(row_i * partition_stride);
      src_sub = ld_tile.reg;
      ctx.update_partition_num(row_i + 1);

      // accum_t scalar_value = src_sub.xetla_select<1, 1>(0)[0];
      // sycl::ext::oneapi::experimental::printf("head_id: %d, sg_id: %d
      // scalar_value: %f\n",
      //     ctx.head_id, ctx.sg_id, scalar_value); // Debugging output

      int32_t remain = ctx.num_partitions - i;
      if (remain < partition_stride) {
        xetla_mask<partition_stride> mask =
            xetla_vector_gen<uint32_t, partition_stride>(1, 1) > remain;
        src_sub.merge(init_value, mask);
      }
    }
  }

  // -------------------- // rescale_exp_sums // -------------------- //

  inline void rescale_exp_sums(tile_t& mat_exp_sums, arguments_t& args) {
    using wg_reduce_max_t =
        group_reduce_t<tile_t, wg_size, reduce_op::max, arch_tag>;

    tile_t mat_max_logits(neg_infinity);
    load_data(mat_max_logits, args.max_logits, args, neg_infinity);

    wg_reduce_max_t wg_reduce_max(
        ctx.num_partition_rows, ctx.sg_id, 0, slm_offset_reduce);
    accum_t global_max_logit = wg_reduce_max(mat_max_logits);
    if constexpr (wg_size > 1) {
      ctx.nbarrier.arrive();
    }

    load_data(mat_exp_sums, args.exp_sums, args, 0);

    for (int i = 0; i < ctx.num_partition_rows; i++) {
      const int offset = i * partition_stride;
      auto max_logits =
          mat_max_logits.reg.xetla_select<partition_stride, 1>(offset);
      auto exp_sums =
          mat_exp_sums.reg.xetla_select<partition_stride, 1>(offset);
      exp_sums = exp_sums * xetla_exp<accum_t>(max_logits - global_max_logit);
    }

    if constexpr (wg_size > 1) {
      ctx.nbarrier.wait();
    }
  }

  // -------------------- // compute_out // -------------------- //

  using out_tile_desc_t = subgroup::tile_desc_t<max_head_size, 1, 16, 1>;
  using out_tile_t = subgroup::tile_t<accum_t, out_tile_desc_t>;

  inline void compute_out(
      tile_t& rescaled_exp_sums,
      out_tile_t& mat_out,
      arguments_t& args) {
    // constexpr uint32_t sg_tile_size_x = partition_stride > 32 /
    // sizeof(scalar_t)
    //     ? 32 / sizeof(scalar_t)
    //     : partition_stride;
    constexpr uint32_t sg_tile_size_x =
        std::min(uint32_t(partition_stride), uint32_t(32 / sizeof(scalar_t)));
    // constexpr uint32_t sg_tile_size_y =
    //     head_size_stride > 16 ? 16 : head_size_stride;
    constexpr uint32_t sg_tile_size_y =
        std::min(uint32_t(head_size_stride), 16u);

    // for loading tmp out to register
    using tmp_tile_desc_t = subgroup::tile_desc_t<
        partition_stride,
        head_size_stride,
        sg_tile_size_x,
        sg_tile_size_y>;
    using tmp_1d_tile_desc_t =
        subgroup::tile_desc_t<head_size_stride, 1, sg_tile_size_y, 1>;
    using tmp_tile_t = subgroup::tile_t<scalar_t, tmp_tile_desc_t>;
    using tmp_1d_tile_t = subgroup::tile_t<scalar_t, tmp_1d_tile_desc_t>;
    using tmp_acc_tile_t = subgroup::tile_t<accum_t, tmp_tile_desc_t>;

    using tmp_prefetch_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::prefetch_payload_t<
            mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>,
            tmp_tile_desc_t,
            1,
            arch_tag>,
        subgroup::prefetch_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            tmp_1d_tile_desc_t,
            1,
            arch_tag>>;

    using tmp_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>,
            tmp_tile_desc_t,
            msg_type::block_2d,
            arch_tag>,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            tmp_1d_tile_desc_t,
            msg_type::block_1d,
            arch_tag>>;
    constexpr tdesc_update_dir tmp_update_dir = tdesc_update_dir::x_dir;

    using wg_reduce_sum_t =
        group_reduce_t<tile_t, wg_size, reduce_op::sum, arch_tag>;

    wg_reduce_sum_t wg_reduce_sum(
        ctx.num_partition_rows, ctx.sg_id, 0, slm_offset_reduce);
    accum_t global_exp_sum = wg_reduce_sum(rescaled_exp_sums);
    accum_t inv_global_exp_sum = 1.0f / global_exp_sum;

    tmp_tile_t mat_tmp_out(0);
    const int loop_count = DIVIDE_ROUND_UP(max_head_size, head_size_stride);

    int32_t base_x =
        (ctx.seq_id * args.num_heads + ctx.head_id) * args.max_num_partitions;
    uint32_t boundary_x = base_x + ctx.num_partitions;
    for (int start_x = ctx.sg_id * partition_stride, row_i = 0;
         start_x < ctx.num_partitions;
         start_x += ctx.wg_partition_stride, row_i++) {
      tmp_payload_t tmp_payload(
          args.tmp_out,
          max_head_size,
          boundary_x,
          max_head_size,
          0,
          base_x + start_x);
      tmp_prefetch_payload_t tmp_prefetch_payload(
          args.tmp_out,
          max_head_size,
          boundary_x,
          max_head_size,
          0,
          base_x + start_x);

#pragma unroll
      for (int i = 0; i < stages; i++) {
        constexpr tdesc_update_dir update_dir =
            has_2d_ld_st ? tmp_update_dir : tdesc_update_dir::y_dir;
        constexpr int update_stride = has_2d_ld_st ? head_size_stride : 1;
        subgroup::tile_prefetch(tmp_prefetch_payload);
        tmp_prefetch_payload.template update_tdesc<update_dir>(update_stride);
      }

      auto rescaled_exp_sums_sub =
          rescaled_exp_sums.reg.xetla_select<partition_stride, 1>(
              row_i * partition_stride);
      rescaled_exp_sums_sub = rescaled_exp_sums_sub * inv_global_exp_sum;

      for (int j = 0; j < loop_count; j++) {
        tile_load_2d<
            tmp_tile_t,
            tmp_payload_t,
            tmp_prefetch_payload_t,
            tmp_update_dir,
            true,
            stages>(
            mat_tmp_out,
            tmp_payload,
            tmp_prefetch_payload,
            max_head_size,
            boundary_x);

        tmp_acc_tile_t mat_tmp_out_acc;
        subgroup::elemwise_cvt(mat_tmp_out_acc, mat_tmp_out);
        auto out_sub =
            mat_out.reg.xetla_select<head_size_stride, 1>(j * head_size_stride);
        out_sub += mat_vec_mul<accum_t, partition_stride, tmp_acc_tile_t, 1>(
            rescaled_exp_sums_sub, mat_tmp_out_acc);
      }
    }
  }

  // -------------------- // compute_out // -------------------- //

  inline void collect_out(out_tile_t& mat_out, arguments_t& args) {
    constexpr uint32_t sg_tile_size_x = head_size_per_sg > 32 / sizeof(scalar_t)
        ? 32 / sizeof(scalar_t)
        : head_size_per_sg;
    constexpr uint32_t sg_tile_size_y = wg_size > 16 ? 16 : wg_size;

    // for storing out to slm
    using local_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        out_tile_desc_t,
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
    using st_tile_desc_t =
        subgroup::tile_desc_t<head_size_per_sg, 1, sg_tile_size_x, 1>;
    using global_st_tile_t = subgroup::tile_t<scalar_t, st_tile_desc_t>;
    using global_st_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            st_tile_desc_t,
            msg_type::block_2d,
            arch_tag>,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            st_tile_desc_t,
            msg_type::block_1d,
            arch_tag>>;

    // store out data of each subgroup to slm
    int32_t start_y = ctx.sg_id;
    local_st_payload_t local_st_payload(
        slm_offset_out, max_head_size, wg_size, max_head_size, 0, start_y);
    subgroup::tile_store(mat_out, local_st_payload);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();

    // load out data to register
    uint32_t boundary_x = max_head_size;
    uint32_t boundary_y = args.num_seqs * args.num_heads;
    uint32_t pitch = max_head_size;
    int32_t start_x = ctx.sg_id * head_size_per_sg;
    start_y = ctx.seq_id * args.num_heads + ctx.head_id;

    if (start_x < max_head_size) {
      // load out data to register
      local_ld_tile_t mat_out_ld;
      local_ld_payload_t local_ld_payload(
          slm_offset_out, max_head_size, wg_size, max_head_size, start_x, 0);
      subgroup::tile_load(mat_out_ld, local_ld_payload);

      // do reduction
      global_st_tile_t mat_out_st;
      mat_out_st.reg =
          subgroup::tile_reduce<reduce_op::sum, scalar_t, accum_t, 0>(
              mat_out_ld);

      // store out to global memory
      global_st_payload_t global_st_payload(
          args.out, boundary_x, boundary_y, pitch, start_x, start_y);
      subgroup::tile_store(mat_out_st, global_st_payload);
    }
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
  static inline sycl::nd_range<3> get_nd_range(
      uint32_t num_seqs,
      uint32_t num_heads) {
    static const sycl::range<3> local_range = sycl::range<3>{1, 1, wg_size};
    sycl::range<3> group_range = sycl::range<3>{num_seqs, num_heads, 1};
    // printf("group_range: %zu, %zu, %zu local_range: %zu, %zu, %zu\n",
    //        group_range[0], group_range[1], group_range[2],
    //        local_range[0], local_range[1], local_range[2]); // Debugging
    //        output */
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
  }
};

template <
    typename policy,
    typename scalar_t,
    typename kv_t,
    typename index_t,
    gpu_arch arch_tag,
    fp8_format fp8_format_ = fp8_format::E4M3>
class paged_attention_loop_kernel {
 public:
  using accum_t = float;

  struct unused_t {};

  struct arguments_t {
    // Input and output tensors
    scalar_t* out; // [num_seqs, num_heads, head_size]
    scalar_t* query; // [num_seqs, num_heads, head_size]
    kv_t* key_cache; // [num_blocks, block_size, num_kv_heads, head_size]
    kv_t* value_cache; // [num_blocks, block_size, num_kv_heads, head_size]
    scalar_t* sinks; // [num_heads]
    float* alibi_slopes; // [num_heads] - alibi_slopes

    // Index
    index_t* block_tables; // [num_seqs, max_blocks_per_seq]
    index_t* context_lens; // [num_seqs]

    // Softmax scale
    accum_t sm_scale;

    // Dimension size
    uint32_t num_seqs;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_size;
    uint32_t max_blocks_per_seq;
    accum_t softcap;
    float fp8_scale;

    inline arguments_t(
        scalar_t* out,
        scalar_t* query,
        kv_t* key_cache,
        kv_t* value_cache,
        scalar_t* sinks,
        float* alibi_slopes,
        index_t* block_tables,
        index_t* context_lens,
        accum_t sm_scale,
        uint32_t num_seqs,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t max_blocks_per_seq,
        accum_t softcap,
        float fp8_scale = 1.0f)
        : out(out),
          query(query),
          key_cache(key_cache),
          value_cache(value_cache),
          sinks(sinks),
          alibi_slopes(alibi_slopes),
          block_tables(block_tables),
          context_lens(context_lens),
          sm_scale(sm_scale),
          num_seqs(num_seqs),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_size(head_size),
          max_blocks_per_seq(max_blocks_per_seq),
          softcap(softcap),
          fp8_scale(fp8_scale) {}
  };

  using tanh_t = typename subgroup::tanh_op_t;

 private:
  // -------------------- // Compute policy // -------------------- //

  static constexpr accum_t neg_infinity = INFINITY * -1;
  static constexpr uint32_t stages = policy::stages;
  static constexpr uint32_t wg_size = policy::wg_size;
  static constexpr uint32_t block_size = policy::block_size;
  static constexpr uint32_t max_head_size = policy::max_head_size;
  static constexpr uint32_t head_size_stride = policy::head_size_stride;
  static constexpr uint32_t max_blocks_per_sg = policy::max_blocks_per_sg;
  static constexpr uint32_t query_group_size = policy::query_group_size;
  /* static constexpr uint32_t num_loop = policy::num_loop; */
  static constexpr uint32_t rows_per_sg =
      DIVIDE_ROUND_UP(query_group_size, wg_size);

  // used for preload query and store output
  // use minimum 16 to avoid mask in load
  static constexpr uint32_t head_size_per_sg =
      std::max(max_head_size / wg_size, 16u);
  static constexpr uint32_t partition_size = policy::partition_size;
  static constexpr bool use_partition = partition_size > 0;
  static constexpr uint32_t value_blocks_per_sg = partition_size / block_size;

  // -------------------- // Slm and nbarrier // -------------------- //

  static constexpr uint32_t slm_size_score =
      query_group_size * partition_size * sizeof(accum_t);
  static constexpr uint32_t slm_size_exp_score =
      query_group_size * partition_size * sizeof(scalar_t);
  static constexpr uint32_t slm_size_rescale_factors =
      query_group_size * sizeof(accum_t);
  static constexpr uint32_t slm_size_exp_sum =
      query_group_size * sizeof(accum_t);

  static constexpr uint32_t slm_offset_score = 0;
  static constexpr uint32_t slm_offset_exp_score =
      slm_offset_score + slm_size_score;
  static constexpr uint32_t slm_offset_rescale_factors =
      slm_offset_exp_score + slm_size_exp_score;
  static constexpr uint32_t slm_offset_exp_sum =
      slm_offset_rescale_factors + slm_size_rescale_factors;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? 1 : 0;
  // This boolean variable will determine whether the kernel will execute 2d
  // load path, the original 2d load paged attention path is disabled by default
  // here to prevent the accuracy issue brought by the surface width limitation
  // of 2d load.
  static constexpr bool has_2d_ld_st =
      (arch_tag == gpu_arch::XeHpc) ? true : false;

  static constexpr int loop_count =
      DIVIDE_ROUND_UP(max_head_size, head_size_stride);

  // -------------------- // Context // -------------------- //

  struct context_t {
    uint32_t sg_id;
    uint32_t seq_id;
    uint32_t head_id;
    uint32_t max_num_partitions;
    uint32_t context_len;
    int kv_head_id;

    index_t* block_table; // [max_blocks_per_seq]
    uint32_t num_blocks_per_sg; // number of blocks processed by current sg

    int kv_block_stride;
    int kv_head_stride;
    int start_block_id;
    int end_block_id;
    uint32_t max_num_blocks;
    uint32_t num_blocks_per_wg;

    float alibi_slopes;
    accum_t sinks_array[query_group_size];

    xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;

    inline context_t() = default;

    inline void init(sycl::nd_item<3>& item, arguments_t& args) {
      sg_id = item.get_local_linear_id();
      kv_head_id = item.get_group(1);
      seq_id = item.get_group(0);

      // head_id = kv_head_id * query_group_size;
      // if (args.alibi_slopes != nullptr) {
      //   alibi_slopes = args.alibi_slopes[head_id];
      // }
      if (args.sinks != nullptr) {
#pragma unroll
        for (uint32_t i = 0; i < query_group_size; i++) {
          sinks_array[i] = static_cast<accum_t>(
              args.sinks[kv_head_id * query_group_size + i]);
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < query_group_size; i++) {
          sinks_array[i] = neg_infinity;
        }
      }

      block_table = args.block_tables + seq_id * args.max_blocks_per_seq;
      num_blocks_per_sg = 0;

      kv_block_stride = args.num_kv_heads * max_head_size * block_size;
      kv_head_stride = max_head_size * kv_head_id;

      context_len = args.context_lens[seq_id];
      max_num_partitions = DIVIDE_ROUND_UP(context_len, partition_size);
      max_num_blocks = DIVIDE_ROUND_UP(context_len, block_size);
      num_blocks_per_wg =
          use_partition ? partition_size / block_size : max_num_blocks;

      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
    }

    inline void update_block_num(uint32_t num) {
      num_blocks_per_sg = num;
    }
  };

  context_t ctx;

  // tile for global max and sum
  using max_logits_tile_desc_t =
      subgroup::tile_desc_t<1, query_group_size, 1, query_group_size>;
  using max_logits_tile_t = subgroup::tile_t<accum_t, max_logits_tile_desc_t>;
  using exp_sum_tile_t = subgroup::tile_t<accum_t, max_logits_tile_desc_t>;

  // -------------------- // compute_score // -------------------- //

  static constexpr uint32_t mma_sg_tile_size = 16;
  using score_tile_desc_t = subgroup::tile_desc_t<
      block_size * max_blocks_per_sg,
      query_group_size,
      mma_sg_tile_size,
      query_group_size>;
  using score_tile_t = subgroup::tile_t<accum_t, score_tile_desc_t>;

  // Compute query x key.
  inline void compute_score(arguments_t& args, const int partition_idx) {
    constexpr uint32_t sg_tile_size_head =
        head_size_stride > 32 / sizeof(scalar_t) ? 32 / sizeof(scalar_t)
                                                 : head_size_stride; // 16
    constexpr uint32_t sg_tile_size_block = mma_sg_tile_size;

    using score_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        score_tile_desc_t,
        msg_type::scatter,
        arch_tag>;

    using query_tile_desc_t = subgroup::tile_desc_t<
        head_size_stride,
        query_group_size,
        sg_tile_size_head,
        query_group_size>;
    using query_tile_t = subgroup::tile_t<scalar_t, query_tile_desc_t>;
    using query_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        query_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    constexpr tdesc_update_dir query_update_dir = tdesc_update_dir::x_dir;
    using key_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        head_size_stride,
        sg_tile_size_block,
        sg_tile_size_head,
        reg_layout::vnni_tiled>;
    constexpr mem_layout k_mem_layout = mem_layout::col_major;
    constexpr tdesc_update_dir key_update_dir = tdesc_update_dir::x_dir;

    using key_tile_t = subgroup::tile_t<kv_t, key_tile_desc_t>;
    using key_mma_tile_t = subgroup::tile_t<scalar_t, key_tile_desc_t>;
    using key_payload_t = subgroup::mem_payload_t<
        mem_desc_t<kv_t, k_mem_layout, mem_space::global>,
        key_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    using score_acc_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        query_group_size,
        mma_sg_tile_size,
        query_group_size>;
    using score_acc_tile_t = subgroup::tile_t<accum_t, score_acc_tile_desc_t>;

    using tile_mma = subgroup::tile_mma_t<
        score_acc_tile_t,
        score_acc_tile_t,
        key_mma_tile_t,
        query_tile_t,
        mma_engine::xmx,
        arch_tag>;

    using dequantize_key_t = std::conditional_t<
        std::is_same_v<kv_t, uint8_t>,
        subgroup::dequant_fp8_weight_t<
            key_mma_tile_t,
            key_tile_t,
            fp8_format_,
            false>,
        unused_t>;

    query_tile_t mat_query;
    key_tile_t mat_key;
    key_mma_tile_t mat_key_mma;
    score_acc_tile_t score_sub(neg_infinity);
    dequantize_key_t dequant_key;

    constexpr uint32_t boundary_score_y = query_group_size;
    uint32_t start_score_x = ctx.sg_id * block_size;
    uint32_t boundary_score_x = start_score_x + block_size;

    score_payload_t score_payload(
        slm_offset_score,
        boundary_score_x,
        boundary_score_y,
        partition_size,
        start_score_x,
        0);

    // iterate over context blocks
    int start_block_idx = partition_idx * ctx.num_blocks_per_wg;
    int end_block_idx =
        std::min(ctx.max_num_blocks, start_block_idx + ctx.num_blocks_per_wg);
    for (int bid = ctx.sg_id + start_block_idx, row_i = 0; bid < end_block_idx;
         bid += wg_size, row_i++) {
      ctx.update_block_num(row_i + 1);

      // get the physical block id from block_table
      // Note, we didn't add correct boundary for context length, as we handled
      // this in following mask
      const int block_id = ctx.block_table[bid];

      int32_t start_y = block_id * block_size;
      uint32_t boundary_y = start_y + block_size;
      int32_t start_x = ctx.kv_head_stride;
      uint32_t boundary_x = start_x + max_head_size;
      uint32_t pitch = max_head_size * args.num_kv_heads;
      auto* cur_key_cache = args.key_cache;
      key_payload_t key_payload(
          cur_key_cache, boundary_x, boundary_y, pitch, start_x, start_y);

      constexpr uint32_t boundary_query_y = query_group_size;
      constexpr uint32_t boundary_query_x = max_head_size;
      auto* cur_query = args.query +
          ctx.seq_id * args.num_heads * max_head_size +
          ctx.kv_head_id * query_group_size * max_head_size;

      query_payload_t query_payload(
          cur_query, boundary_query_x, boundary_query_y, max_head_size, 0, 0);

      score_sub.reg = 0;
#pragma unroll
      for (int i = 0; i < loop_count; i++) {
        subgroup::tile_load(mat_query, query_payload);
        subgroup::tile_load(mat_key, key_payload);

        query_payload.template update_tdesc<query_update_dir>(head_size_stride);
        key_payload.template update_tdesc<key_update_dir>(head_size_stride);

        if constexpr (std::is_same_v<kv_t, uint8_t>) {
          dequant_key(mat_key_mma, mat_key, args.fp8_scale);
        } else {
          mat_key_mma = mat_key;
        }

        SW_BARRIER();
        tile_mma::mma(score_sub, score_sub, mat_key_mma, mat_query);
        SW_BARRIER();
      }
      score_sub.reg *= args.sm_scale;

      if constexpr (std::is_same_v<kv_t, uint8_t>) {
        score_sub.reg *= args.fp8_scale;
      }

      uint32_t remained_len = ctx.context_len - bid * block_size;
      using remain_tile_mask = fmha::tile_mask_t<score_acc_tile_t>;
      if (remained_len < block_size) {
        remain_tile_mask::padding_mask(score_sub, remained_len);
      }

      // if (args.softcap > 0.0) {
      //   score_sub.reg /= args.softcap;
      //   tanh_t tanh;
      //   tanh(score_sub, 0);
      //   score_sub.reg *= args.softcap;
      // }

      // if (args.alibi_slopes != nullptr) {
      //   int32_t mat_real_x = bid * block_size;
      //   int32_t mat_real_y = ctx.seq_id;
      //   xetla_vector<float, block_size> pos_id =
      //       xetla_vector_gen<float, block_size>(mat_real_x, 1);
      //   score_sub += (pos_id * ctx.alibi_slopes);
      // }
    }
    subgroup::tile_store(score_sub, score_payload);
    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

  // -------------------- // softmax // -------------------- //

  // Compute softmax of score.
  inline void softmax(
      arguments_t& args,
      accum_t (&max_logits_old)[rows_per_sg],
      accum_t (&exp_sums_old)[rows_per_sg],
      const int partition_idx) {
    for (int row_id = ctx.sg_id, row_sg = 0; row_id < query_group_size;
         row_id += wg_size, row_sg++) {
      using softmax_score_tile_desc_t =
          subgroup::tile_desc_t<partition_size, 1, mma_sg_tile_size, 1>;
      using softmax_score_tile_t =
          subgroup::tile_t<accum_t, softmax_score_tile_desc_t>;
      softmax_score_tile_t sm_mat_score;

      using sm_local_ld_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
          softmax_score_tile_desc_t,
          msg_type::block_1d,
          arch_tag>;
      using sm_local_st_payload_t = subgroup::mem_payload_t<
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
          softmax_score_tile_desc_t,
          msg_type::block_1d,
          arch_tag>;

      uint32_t start_y = row_id;
      constexpr uint32_t boundary_y = query_group_size;
      sm_local_ld_payload_t sm_local_ld_payload(
          slm_offset_score,
          partition_size,
          boundary_y,
          partition_size,
          0,
          start_y);

      subgroup::tile_load(sm_mat_score, sm_local_ld_payload);
      xetla_vector<accum_t, 1> max_score =
          subgroup::tile_reduce<reduce_op::max, accum_t, accum_t, 1>(
              sm_mat_score);

      accum_t sink =
          partition_idx == 0 ? ctx.sinks_array[row_id] : neg_infinity;
      accum_t scalar_max = max_score[0];
      scalar_max = std::max(scalar_max, sink);
      accum_t max_logits_cur = std::max(max_logits_old[row_sg], scalar_max);
      sm_mat_score.reg -= max_logits_cur;
      sm_mat_score.reg = xetla_exp<accum_t>(sm_mat_score.reg);
      sink = std::exp(sink - max_logits_cur);

      xetla_vector<accum_t, 1> sum_score =
          subgroup::tile_reduce<reduce_op::sum, accum_t, accum_t, 1>(
              sm_mat_score);
      accum_t scalar_sum = sum_score[0];
      scalar_sum += sink;

      // Store the softmax result back to shared local memory
      using softmax_exp_score_tile_t =
          subgroup::tile_t<scalar_t, softmax_score_tile_desc_t>;
      softmax_exp_score_tile_t sm_mat_exp_score;
      subgroup::elemwise_cvt(sm_mat_exp_score, sm_mat_score);

      sm_local_st_payload_t sm_local_st_payload(
          slm_offset_exp_score,
          partition_size,
          boundary_y,
          partition_size,
          0,
          start_y);
      subgroup::tile_store(sm_mat_exp_score, sm_local_st_payload);

      // update max and sum
      accum_t rescale_factor =
          std::exp(max_logits_old[row_sg] - max_logits_cur);
      accum_t exp_sums_cur = rescale_factor * exp_sums_old[row_sg] + scalar_sum;
      max_logits_old[row_sg] = max_logits_cur;
      exp_sums_old[row_sg] = exp_sums_cur;

      // store rescale_factor and exp_sum into shared local memory for
      // compute_out
      using rescale_expsum_tile_desc_t = subgroup::tile_desc_t<1, 1, 1, 1>;
      using rescale_expsum_tile_t =
          subgroup::tile_t<accum_t, rescale_expsum_tile_desc_t>;
      using rescale_expsum_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
          rescale_expsum_tile_desc_t,
          msg_type::block_1d,
          arch_tag>;

      rescale_expsum_tile_t mat_rescale(rescale_factor);
      rescale_expsum_payload_t rescale_factor_payload(
          slm_offset_rescale_factors, 1, query_group_size, 1, 0, row_id);
      subgroup::tile_store(mat_rescale, rescale_factor_payload);

      rescale_expsum_tile_t mat_exp_sum(exp_sums_old[row_sg]);
      rescale_expsum_payload_t exp_sum_payload(
          slm_offset_exp_sum, 1, query_group_size, 1, 0, row_id);
      subgroup::tile_store(mat_exp_sum, exp_sum_payload);
    } // end of for each row in query_group_size

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

  // -------------------- // compute_out // -------------------- //

  using out_tile_desc_t = subgroup::tile_desc_t<
      head_size_per_sg,
      query_group_size,
      head_size_per_sg,
      query_group_size>;
  using out_acc_tile_t = subgroup::tile_t<accum_t, out_tile_desc_t>;
  using out_tile_t = subgroup::tile_t<scalar_t, out_tile_desc_t>;

  // Compute output.
  inline void compute_out(
      arguments_t& args,
      out_acc_tile_t& mat_acc_old,
      const int partition_idx) {
    constexpr uint32_t sg_tile_size_head = std::min(
        uint32_t(head_size_stride), uint32_t(32 / sizeof(scalar_t))); // 16
    constexpr uint32_t sg_tile_size_block =
        std::min(uint32_t(block_size), 32u); // 32
    constexpr uint32_t sg_tile_size_x =
        std::min(uint32_t(block_size), uint32_t(32 / sizeof(scalar_t))); // 16
    constexpr uint32_t sg_tile_size_y = 16;

    using value_tile_desc_t = subgroup::tile_desc_t<
        head_size_per_sg,
        block_size,
        sg_tile_size_x,
        sg_tile_size_y,
        reg_layout::vnni_tiled>;
    using value_tile_t = subgroup::tile_t<kv_t, value_tile_desc_t>;
    using value_mma_tile_t = subgroup::tile_t<scalar_t, value_tile_desc_t>;
    using value_payload_t = subgroup::mem_payload_t<
        mem_desc_t<kv_t, mem_layout::row_major, mem_space::global>,
        value_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    using exp_score_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        query_group_size,
        sg_tile_size_head,
        query_group_size>;
    using exp_score_tile_t = subgroup::tile_t<scalar_t, exp_score_tile_desc_t>;
    using exp_score_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
        exp_score_tile_desc_t,
        msg_type::scatter,
        arch_tag>;
    constexpr tdesc_update_dir exp_score_update_dir = tdesc_update_dir::x_dir;

    constexpr uint32_t boundary_e_y = query_group_size;
    constexpr uint32_t boundary_e_x = block_size;
    constexpr uint32_t pitch_e = partition_size;
    exp_score_payload_t exp_score_payload(
        slm_offset_exp_score, boundary_e_x, boundary_e_y, pitch_e, 0, 0);

    using tile_mma = subgroup::tile_mma_t<
        out_acc_tile_t,
        out_acc_tile_t,
        value_mma_tile_t,
        exp_score_tile_t,
        mma_engine::xmx,
        arch_tag>;

    using dequant_value_t = std::conditional_t<
        std::is_same_v<kv_t, uint8_t>,
        subgroup::dequant_fp8_weight_t<
            value_mma_tile_t,
            value_tile_t,
            fp8_format_,
            false>,
        unused_t>;

    exp_score_tile_t mat_exp_score;
    value_tile_t mat_value;
    value_mma_tile_t mat_value_mma;
    out_acc_tile_t cur_mat_acc_out(0);
    dequant_value_t dequant_value;

    const int start_block_idx = partition_idx * ctx.num_blocks_per_wg;
    const int end_block_idx =
        std::min(ctx.max_num_blocks, start_block_idx + ctx.num_blocks_per_wg);

#pragma unroll(value_blocks_per_sg)
    for (uint32_t i = 0; i < value_blocks_per_sg; ++i) {
      uint32_t cur_bid = i + start_block_idx;

      if (cur_bid >= end_block_idx)
        break;

      uint32_t block_id = ctx.block_table[cur_bid];

      constexpr uint32_t boundary_v_y = block_size;
      uint32_t start_v_x =
          ctx.kv_head_id * max_head_size + ctx.sg_id * head_size_per_sg;
      uint32_t boundary_v_x = start_v_x + head_size_per_sg;
      uint32_t pitch_v = args.num_kv_heads * max_head_size;
      auto* cur_value_cache = args.value_cache + block_id * ctx.kv_block_stride;
      value_payload_t value_payload(
          cur_value_cache, boundary_v_x, boundary_v_y, pitch_v, start_v_x, 0);

      subgroup::tile_load(mat_exp_score, exp_score_payload);
      subgroup::tile_load(mat_value, value_payload);

      exp_score_payload.template update_tdesc<exp_score_update_dir>(block_size);

      if constexpr (std::is_same_v<kv_t, uint8_t>) {
        dequant_value(mat_value_mma, mat_value, args.fp8_scale);
      } else {
        mat_value_mma = mat_value;
      }

      SW_BARRIER();
      tile_mma::mma(
          cur_mat_acc_out, cur_mat_acc_out, mat_value_mma, mat_exp_score);
      SW_BARRIER();
    }

    if constexpr (std::is_same_v<kv_t, uint8_t>) {
      cur_mat_acc_out.reg *= args.fp8_scale;
    }

    // load rescale_factor from slm
    using rescale_expsum_load_tile_desc_t =
        subgroup::tile_desc_t<1, query_group_size, 1, query_group_size>;
    using rescale_expsum_load_tile_t =
        subgroup::tile_t<accum_t, rescale_expsum_load_tile_desc_t>;
    using rescale_expsum_load_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        rescale_expsum_load_tile_desc_t,
        msg_type::block_1d,
        arch_tag>;

    rescale_expsum_load_payload_t rescale_load_payload(
        slm_offset_rescale_factors, 1, query_group_size, 1, 0, 0);
    rescale_expsum_load_tile_t mat_rescale;
    subgroup::tile_load(mat_rescale, rescale_load_payload);
    rescale_expsum_load_tile_t mat_exp_sum;
    rescale_expsum_load_payload_t exp_sum_load_payload(
        slm_offset_exp_sum, 1, query_group_size, 1, 0, 0);
    subgroup::tile_load(mat_exp_sum, exp_sum_load_payload);

    mat_acc_old.reg =
        mat_vec_mul_broadcast<accum_t, query_group_size, out_acc_tile_t, 0>(
            mat_rescale.reg, mat_acc_old) +
        cur_mat_acc_out.reg;
  }

  // -------------------- // store_out // -------------------- //

  inline void store_out(arguments_t& args, out_acc_tile_t& mat_acc_out) {
    using rescale_expsum_load_tile_desc_t =
        subgroup::tile_desc_t<1, query_group_size, 1, query_group_size>;
    using rescale_expsum_load_tile_t =
        subgroup::tile_t<accum_t, rescale_expsum_load_tile_desc_t>;
    using rescale_expsum_load_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>,
        rescale_expsum_load_tile_desc_t,
        msg_type::block_1d,
        arch_tag>;

    rescale_expsum_load_payload_t exp_sum_load_payload(
        slm_offset_exp_sum, 1, query_group_size, 1, 0, 0);
    rescale_expsum_load_tile_t mat_exp_sum;
    subgroup::tile_load(mat_exp_sum, exp_sum_load_payload);

    mat_acc_out.reg =
        mat_vec_div_broadcast<accum_t, query_group_size, out_acc_tile_t, 0>(
            mat_exp_sum.reg, mat_acc_out);

    out_tile_t mat_out;
    subgroup::elemwise_cvt(mat_out, mat_acc_out);

    using out_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        out_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;

    uint32_t start_o_x = ctx.sg_id * head_size_per_sg;
    uint32_t boundary_o_x = start_o_x + head_size_per_sg;
    constexpr uint32_t boundary_o_y = query_group_size;
    constexpr uint32_t pitch_o = max_head_size;
    auto* cur_out = args.out + ctx.seq_id * args.num_heads * max_head_size +
        ctx.kv_head_id * query_group_size * max_head_size;

    out_st_payload_t out_st_payload(
        cur_out, boundary_o_x, boundary_o_y, pitch_o, start_o_x, 0);

    subgroup::tile_store(mat_out, out_st_payload);
  }

 public:
  // Get the local memory size consumption.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_score + slm_size_exp_score +
        slm_size_exp_sum + slm_size_rescale_factors;
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
  static inline sycl::nd_range<3> get_nd_range(
      uint32_t num_seqs,
      uint32_t num_kv_heads) {
    static const sycl::range<3> local_range = sycl::range<3>{1, 1, wg_size};
    sycl::range<3> group_range = sycl::range<3>{num_seqs, num_kv_heads, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  // Entrance of the functor
  inline SYCL_ESIMD_FUNCTION void operator()(
      sycl::nd_item<3>& item,
      arguments_t& args) {
    // initialization

    ctx.init(item, args);

    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // global max and sum
    accum_t max_logits_old[rows_per_sg];
    accum_t exp_sums_old[rows_per_sg];
#pragma unroll
    for (int i = 0; i < rows_per_sg; i++) {
      max_logits_old[i] = neg_infinity;
      exp_sums_old[i] = 0.0f;
    }

    out_acc_tile_t mat_out_acc(0.0f);

    for (int p_id = 0; p_id < ctx.max_num_partitions; p_id++) {
      compute_score(args, p_id);
      softmax(args, max_logits_old, exp_sums_old, p_id);
      compute_out(args, mat_out_acc, p_id);
    }
    store_out(args, mat_out_acc);
  }
};

#undef DIVIDE_ROUND_UP

} // namespace attention

} // namespace gpu::xetla
