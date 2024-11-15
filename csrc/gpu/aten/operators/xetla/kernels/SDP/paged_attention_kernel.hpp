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
#include "paged_attention_utils.hpp"
#include "xetla.hpp"

#include <sycl/sycl.hpp>

namespace gpu::xetla {

namespace attention {

#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

/// @brief This function loads data from 1d memory surface with boundary check.
/// The data exceeds the memory boundary will automatically set to 0
/// @tparam mem_payload_t The type of memory payload
/// @tparam tile_t The type of tile
/// @tparam x_contiguous Whether the x axis of the tile is contiguous

template <typename mem_payload_t, typename tile_t, bool x_contiguous = false>
inline std::enable_if_t<x_contiguous == false> tile_load_1d(
    tile_t& mat_load,
    mem_payload_t& mem_payload,
    int boundary_x,
    int boundary_y) {
  using dtype = mem_payload_t::dtype;
  int pitch = mem_payload.pitch_in_bytes / sizeof(dtype);
  int64_t base_offset = mem_payload.base_offset / sizeof(dtype);
  int64_t x_base_offset = base_offset % pitch;
  int64_t y_base_offset = base_offset / pitch;
  int64_t dist_to_x = x_base_offset - boundary_x;
  int64_t dist_to_y = y_base_offset - boundary_y;
  if (dist_to_y >= 0) {
    mat_load.reg = 0;
    return;
  }
  if (dist_to_x >= 0) {
    mat_load.reg = 0;
    return;
  }
  tile_t mask;
  mask.reg = xetla_vector_gen<dtype, tile_t::tile_desc::tile_elems>(0, 1) <
      (boundary_x - x_base_offset);
  subgroup::tile_load(mat_load, mem_payload);
  mat_load.reg *= mask.reg;
}

/// @brief This function loads data from 1d memory surface with boundary check.
/// The data exceeds the memory boundary will automatically set to 0, This
/// function only apply to the x-axis contiguous and no-padding scenario
/// @tparam mem_payload_t The type of memory payload
/// @tparam tile_t The type of tile
/// @tparam x_contiguous Whether the x axis of the tile is contiguous

template <typename mem_payload_t, typename tile_t, bool x_contiguous = false>
inline std::enable_if_t<x_contiguous == true> tile_load_1d(
    tile_t& mat_load,
    mem_payload_t& mem_payload,
    int boundary_x,
    int boundary_y) {
  using dtype = mem_payload_t::dtype;
  int pitch = mem_payload.pitch_in_bytes / sizeof(dtype);
  int64_t base_offset = mem_payload.base_offset / sizeof(dtype);
  int64_t boundary_1d = boundary_y * pitch + boundary_x;
  int64_t dist_to_boundary = base_offset - boundary_1d;
  if (dist_to_boundary >= 0) {
    mat_load.reg = 0;
    return;
  }
  tile_t mask;
  mask.reg = xetla_vector_gen<dtype, tile_t::tile_desc::tile_elems>(0, 1) <
      (boundary_1d - base_offset);
  subgroup::tile_load(mat_load, mem_payload);
  mat_load.reg *= mask.reg;
}

/// @brief This function loads data from 1d memory surface and fill the 2d tile
/// in place. memory prefetch and the pointer for memory payload will update
/// automatically. This functional should only be used inside the paged
/// attention kernel. Cause we assume that the 2d tile's x axis should be
/// contiguous and the x axis size should equals to the element size of real
/// data in that dimension.
/// @tparam tile_type The type of tile
/// @tparam mem_payload_t The type of memory payload
/// @tparam prefetch_payload_t The type of prefetch payload
/// @dir The direction of tile descriptor update
/// @is_transposed Whether the 2d tile is transposed
/// @stages The number of stages for prefetch

template <
    typename tile_type,
    typename mem_payload_type,
    typename prefetch_payload_type,
    tdesc_update_dir dir,
    bool is_transposed,
    int stages>
inline std::enable_if_t<
    mem_payload_type::message_type == msg_type::block_1d &&
    dir == tdesc_update_dir::x_dir>
tile_load_2d(
    tile_type& mat_load,
    mem_payload_type& mem_payload,
    prefetch_payload_type& prefetch_payload,
    int boundary_x,
    int boundary_y) {
  using tile_desc_2d_t = tile_type::tile_desc;
  using tile_desc_1d_t = mem_payload_type::tile_desc;
  static_assert(
      (tile_desc_1d_t::tile_size_x == tile_desc_2d_t::tile_size_x &&
       !is_transposed) ||
          (tile_desc_1d_t::tile_size_x == tile_desc_2d_t::tile_size_y &&
           is_transposed),
      "2d_tile and 1d_tile should have the same tile size");
  using dtype = tile_type::dtype;
  constexpr int loop_cnt =
      is_transposed ? tile_desc_2d_t::tile_size_x : tile_desc_2d_t::tile_size_y;
  constexpr int read_cnt =
      is_transposed ? tile_desc_2d_t::tile_size_y : tile_desc_2d_t::tile_size_x;
  constexpr int view_stride = is_transposed ? tile_desc_2d_t::tile_size_x : 1;
  constexpr int view_offset = is_transposed ? 1 : tile_desc_2d_t::tile_size_x;

  using tile_1d_t = subgroup::tile_t<dtype, tile_desc_1d_t>;
  tile_1d_t mat_1d(0);
#pragma unroll
  for (int i = 0; i < loop_cnt; ++i) {
    auto tile_slice_2d =
        mat_load.reg.xetla_select<read_cnt, view_stride>(i * view_offset);
    tile_load_1d(mat_1d, mem_payload, boundary_x, boundary_y);
    tile_slice_2d = mat_1d.reg;
    mem_payload.template update_tdesc<tdesc_update_dir::y_dir>(1);
    if constexpr (stages != 0) {
      subgroup::tile_prefetch(prefetch_payload);
      prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(1);
    }
  }
  mem_payload.template update_tdesc<tdesc_update_dir::y_dir>(-loop_cnt);
  mem_payload.template update_tdesc<tdesc_update_dir::x_dir>(read_cnt);
  if constexpr (stages != 0) {
    prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(
        -loop_cnt - stages);
    prefetch_payload.template update_tdesc<tdesc_update_dir::x_dir>(read_cnt);
  }
  if constexpr (stages != 0) {
#pragma unroll
    for (int i = 0; i < stages; ++i) {
      subgroup::tile_prefetch(prefetch_payload);
      prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(1);
    }
  }
}

/// @brief This function loads data from 1d memory surface and fill the 2d tile
/// in place. memory prefetch and the pointer for memory payload will update
/// automatically. This functional should only be used inside the paged
/// attention kernel. Cause we assume that the 2d tile's x axis should be
/// contiguous and the x axis size should equals to the element size of real dat
/// in that dimension.
/// @tparam tile_type The type of tile
/// @tparam mem_payload_t The type of memory payload
/// @tparam prefetch_payload_t The type of prefetch payload
/// @dir The direction of tile descriptor update
/// @is_transposed Whether the 2d tile is transposed, it is supposed to be false
/// in this template function
/// @stages The number of stages for prefetch

template <
    typename tile_type,
    typename mem_payload_type,
    typename prefetch_payload_type,
    tdesc_update_dir dir,
    bool is_transposed,
    int stages>
inline std::enable_if_t<
    mem_payload_type::message_type == msg_type::block_1d &&
    dir == tdesc_update_dir::y_dir && is_transposed == false>
tile_load_2d(
    tile_type& mat_load,
    mem_payload_type& mem_payload,
    prefetch_payload_type& prefetch_payload,
    int boundary_x,
    int boundary_y) {
  using tile_desc_1d_t = mem_payload_type::tile_desc;
  using dtype = tile_type::dtype;
  using tile_1d_t = subgroup::tile_t<dtype, tile_desc_1d_t>;
  static_assert(
      tile_desc_1d_t::tile_size_x % tile_type::tile_desc::tile_size_x == 0,
      "1d tile's size should be the multiplyer of 2d tile's size at x axis.");
  static_assert(
      tile_type::tile_desc::tile_elems % tile_desc_1d_t::tile_size_x == 0,
      "the elements size in 1d tile should be divisible to 2d tile's element size");
  constexpr uint32_t loop_cnt =
      tile_type::tile_desc::tile_elems / tile_desc_1d_t::tile_size_x;
  tile_1d_t mat_1d;
  constexpr uint32_t y_stride = tile_type::tile_desc::tile_size_y / loop_cnt;
#pragma unroll
  for (int i = 0; i < loop_cnt; ++i) {
    auto tile_slice_2d =
        mat_load.reg.xetla_select<tile_desc_1d_t::tile_size_x, 1>(
            i * tile_desc_1d_t::tile_size_x);
    tile_load_1d<mem_payload_type, tile_1d_t, true>(
        mat_1d, mem_payload, boundary_x, boundary_y);
    tile_slice_2d = mat_1d.reg;
    if constexpr (stages != 0) {
      subgroup::tile_prefetch(prefetch_payload);
    }
    mem_payload.template update_tdesc<tdesc_update_dir::y_dir>(y_stride);
    if constexpr (stages != 0) {
      prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(y_stride);
    }
  }
}

/// @brief This function loads data from 2d memory surface and fill the 2d tile
/// in place. This function will perform vallina 2d load.
/// @tparam tile_type The type of tile
/// @tparam mem_payload_t The type of memory payload
/// @tparam prefetch_payload_t The type of prefetch payload
/// @dir The direction of tile descriptor update
/// @is_transposed Whether the 2d tile is transposed, it is supposed to be false
/// in this template function
/// @stages The number of stages for prefetch
template <
    typename tile_type,
    typename mem_payload_type,
    typename prefetch_payload_type,
    tdesc_update_dir dir,
    bool is_transposed,
    int stages>
inline std::enable_if_t<mem_payload_type::message_type == msg_type::block_2d>
tile_load_2d(
    tile_type& mat_load,
    mem_payload_type& mem_payload,
    prefetch_payload_type& prefetch_payload,
    int boundary_x,
    int boundary_y) {
  constexpr int x_stride =
      is_transposed ? tile_type::tile_size_y : tile_type::tile_size_x;
  constexpr int y_stride =
      is_transposed ? tile_type::tile_size_x : tile_type::tile_size_y;
  constexpr int update_stride =
      dir == tdesc_update_dir::x_dir ? x_stride : y_stride;
  subgroup::tile_load(mat_load, mem_payload);
  if constexpr (stages != 0) {
    subgroup::tile_prefetch(prefetch_payload);
  }
  mem_payload.template update_tdesc<dir>(update_stride);
  if constexpr (stages != 0) {
    prefetch_payload.template update_tdesc<dir>(update_stride);
  }
}

template <
    typename policy,
    typename scalar_t,
    typename index_t,
    gpu_arch arch_tag>
class paged_attention_kernel {
 public:
  using accum_t = float;

  struct arguments_t {
    // Input and output tensors
    accum_t* max_logits; // [num_seqs, num_heads, max_num_partitions]
    accum_t* exp_sums; // [num_seqs, num_heads, max_num_partitions]
    scalar_t* out; // [num_seqs, num_heads, max_num_partitions, head_size]
    scalar_t* query; // [num_seqs, num_heads, head_size]
    scalar_t* key_cache; // [num_blocks, num_kv_heads, head_size, block_size]
    scalar_t* value_cache; // [num_blocks, num_kv_heads, head_size, block_size]

    // Index
    index_t* block_tables; // [num_seqs, max_blocks_per_seq]
    index_t* context_lens; // [num_seqs]

    // The number of queries to process per key/value header
    uint32_t num_queries_per_tokens;

    // Softmax scale
    accum_t sm_scale;

    // Dimension size
    uint32_t num_seqs;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_size;
    uint32_t max_blocks_per_seq;

    inline arguments_t(
        accum_t* max_logits,
        accum_t* exp_sums,
        scalar_t* out,
        scalar_t* query,
        scalar_t* key_cache,
        scalar_t* value_cache,
        index_t* block_tables,
        index_t* context_lens,
        uint32_t num_queries_per_tokens,
        accum_t sm_scale,
        uint32_t num_seqs,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t max_blocks_per_seq)
        : max_logits(max_logits),
          exp_sums(exp_sums),
          out(out),
          query(query),
          key_cache(key_cache),
          value_cache(value_cache),
          block_tables(block_tables),
          context_lens(context_lens),
          num_queries_per_tokens(num_queries_per_tokens),
          sm_scale(sm_scale),
          num_seqs(num_seqs),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_size(head_size),
          max_blocks_per_seq(max_blocks_per_seq) {}
  };

 private:
  // -------------------- // Compute policy // -------------------- //

  static constexpr accum_t neg_infinity = INFINITY * -1;
  static constexpr uint32_t stages = policy::stages;
  static constexpr uint32_t wg_size = policy::wg_size;
  static constexpr uint32_t block_size = policy::block_size;
  static constexpr uint32_t max_head_size = policy::max_head_size;
  static constexpr uint32_t head_size_stride = policy::head_size_stride;
  static constexpr uint32_t max_blocks_per_sg = policy::max_blocks_per_sg;
  // used for preload query and store output
  // use minimum 16 to avoid mask in load
  static constexpr uint32_t head_size_per_sg =
      std::max(max_head_size / wg_size, 16u);
  static constexpr uint32_t partition_size = policy::partition_size;
  static constexpr bool use_partition = partition_size > 0;

  // -------------------- // Slm and nbarrier // -------------------- //

  static constexpr uint32_t slm_size_query = max_head_size * sizeof(scalar_t);
  static constexpr uint32_t slm_size_softmax =
      (wg_size > 1) ? wg_size * sizeof(accum_t) : 0;
  static constexpr uint32_t slm_size_out =
      max_head_size * wg_size * sizeof(accum_t);

  static constexpr uint32_t slm_offset_query = 0;
  static constexpr uint32_t slm_offset_softmax =
      slm_offset_query + slm_size_query;
  static constexpr uint32_t slm_offset_out =
      slm_offset_softmax + slm_size_softmax;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? 1 : 0;
  // This boolean variable will determine whether the kernel will execute 2d
  // load path, the original 2d load paged attention path is disabled by default
  // here to prevent the accuracy issue brought by the surface width limitation
  // of 2d load.
  static constexpr bool has_2d_ld_st = false;

  // -------------------- // Context // -------------------- //

  struct context_t {
    uint32_t sg_id;
    uint32_t seq_id;
    uint32_t head_id;
    uint32_t partition_id;
    uint32_t max_num_partitions;
    uint32_t context_len;

    index_t* block_table; // [max_blocks_per_seq]
    uint32_t num_blocks_per_sg; // number of blocks processed by current sg

    int kv_block_stride;
    int kv_head_stride;
    int start_block_id;
    int end_block_id;
    int loop_count;

    xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;

    inline context_t() = default;

    inline void init(sycl::nd_item<3>& item, arguments_t& args) {
      sg_id = item.get_local_linear_id();
      head_id = item.get_group(0);
      seq_id = item.get_group(1);
      partition_id = item.get_group(2);
      max_num_partitions = item.get_group_range(2);

      context_len = args.context_lens[seq_id];
      block_table = args.block_tables + seq_id * args.max_blocks_per_seq;
      num_blocks_per_sg = 0;

      //   const int kv_head_id = args.head_mapping[head_id];
      const int kv_head_id = head_id / args.num_queries_per_tokens;
      kv_block_stride = args.num_kv_heads * args.head_size;
      kv_head_stride = args.head_size * kv_head_id;

      const int max_num_blocks = DIVIDE_ROUND_UP(context_len, block_size);
      const int num_blocks_per_wg =
          use_partition ? partition_size / block_size : max_num_blocks;

      start_block_id = partition_id * num_blocks_per_wg;
      end_block_id =
          std::min(max_num_blocks, start_block_id + num_blocks_per_wg);

      loop_count = DIVIDE_ROUND_UP(args.head_size, head_size_stride);

      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
    }

    inline void update_block_num(uint32_t num) {
      num_blocks_per_sg = num;
    }
  };

  context_t ctx;

  // -------------------- // preload_query // -------------------- //

  // Pre-load query from global memory to shared local memory.
  inline void preload_query(arguments_t& args) {
    using tile_desc_t = subgroup::tile_desc_t<head_size_per_sg, 1, 16, 1>;
    using tile_t = subgroup::tile_t<scalar_t, tile_desc_t>;

    using global_ld_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        tile_desc_t,
        msg_type::block_1d,
        arch_tag>;
    using local_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
        tile_desc_t,
        msg_type::block_1d,
        arch_tag>;

    uint32_t boundary_x = args.head_size;
    uint32_t boundary_y = args.num_seqs * args.num_heads;
    uint32_t pitch = args.head_size;
    int32_t start_x = ctx.sg_id * head_size_per_sg;
    int32_t start_y = ctx.seq_id * args.num_heads + ctx.head_id;

    if (start_x < args.head_size) {
      global_ld_payload_t ld_payload(
          args.query, boundary_x, boundary_y, pitch, start_x, start_y);
      local_st_payload_t st_payload(
          slm_offset_query, max_head_size, 1, max_head_size, start_x, 0);

      tile_t mat_query;
      subgroup::tile_load(mat_query, ld_payload);
      subgroup::tile_store(mat_query, st_payload);
    }
    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

  // -------------------- // compute_score // -------------------- //

  using score_tile_desc_t = subgroup::
      tile_desc_t<block_size, max_blocks_per_sg, block_size, max_blocks_per_sg>;
  using score_tile_t = subgroup::tile_t<accum_t, score_tile_desc_t>;

  // Compute query x key.
  inline void compute_score(score_tile_t& mat_score, arguments_t& args) {
    constexpr uint32_t sg_tile_size_head =
        head_size_stride > 32 / sizeof(scalar_t) ? 32 / sizeof(scalar_t)
                                                 : head_size_stride;
    constexpr uint32_t sg_tile_size_block = block_size > 32 ? 32 : block_size;

    using query_tile_desc_t =
        subgroup::tile_desc_t<head_size_stride, 1, sg_tile_size_head, 1>;
    using query_tile_t = subgroup::tile_t<scalar_t, query_tile_desc_t>;
    using query_acc_tile_t = subgroup::tile_t<accum_t, query_tile_desc_t>;
    using query_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>,
        query_tile_desc_t,
        msg_type::block_1d,
        arch_tag>;
    constexpr tdesc_update_dir query_update_dir = tdesc_update_dir::x_dir;
    constexpr uint32_t tile_size_1d = arch_tag >= gpu_arch::XeHpc
        ? block_size * head_size_stride
        : block_size;
    using key_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        head_size_stride,
        sg_tile_size_block,
        sg_tile_size_head>;
    constexpr mem_layout k_mem_layout = mem_layout::row_major;
    constexpr tdesc_update_dir key_update_dir = tdesc_update_dir::y_dir;
    using key_1d_tile_desc_t =
        subgroup::tile_desc_t<tile_size_1d, 1, sg_tile_size_block, 1>;
    using key_tile_t = subgroup::tile_t<scalar_t, key_tile_desc_t>;
    using key_1d_tile_t = subgroup::tile_t<scalar_t, key_1d_tile_desc_t>;
    using key_acc_tile_t = subgroup::tile_t<accum_t, key_tile_desc_t>;
    using key_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, k_mem_layout, mem_space::global>,
            key_tile_desc_t,
            msg_type::block_2d,
            arch_tag>,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, k_mem_layout, mem_space::global>,
            key_1d_tile_desc_t,
            msg_type::block_1d,
            arch_tag>>;

    using key_prefetch_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::prefetch_payload_t<
            mem_desc_t<scalar_t, k_mem_layout, mem_space::global>,
            key_tile_desc_t,
            1,
            arch_tag>,
        subgroup::prefetch_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            key_1d_tile_desc_t,
            1,
            arch_tag>>;

    query_tile_t mat_query;
    key_tile_t mat_key;

    // iterate over context blocks
    for (int bid = ctx.sg_id + ctx.start_block_id, row_i = 0;
         bid < ctx.end_block_id;
         bid += wg_size, row_i++) {
      ctx.update_block_num(row_i + 1);

      // get the physical block id from block_table
      // Note, we didn't add correct boundary for context length, as we handled
      // this in following mask
      const int block_id = ctx.block_table[bid];
      int32_t start_y = block_id * ctx.kv_block_stride + ctx.kv_head_stride;
      uint32_t boundary_y = start_y + args.head_size;
      constexpr uint32_t boundary_x = block_size;
      key_payload_t key_payload(
          args.key_cache, boundary_x, boundary_y, block_size, 0, start_y);
      key_prefetch_payload_t key_prefetch_payload(
          args.key_cache, boundary_x, boundary_y, block_size, 0, start_y);

      query_payload_t query_payload(
          slm_offset_query, max_head_size, 1, max_head_size, 0, 0);

#pragma unroll
      for (int i = 0; i < stages; i++) {
        constexpr int key_update_stride = has_2d_ld_st ? head_size_stride : 1;
        subgroup::tile_prefetch(key_prefetch_payload);
        key_prefetch_payload.template update_tdesc<key_update_dir>(
            key_update_stride);
      }

      auto score_sub =
          mat_score.reg.xetla_select<block_size, 1>(row_i * block_size);

      for (int i = 0; i < ctx.loop_count; i++) {
        subgroup::tile_load(mat_query, query_payload);
        constexpr bool is_transposed = k_mem_layout == mem_layout::col_major;
        tile_load_2d<
            key_tile_t,
            key_payload_t,
            key_prefetch_payload_t,
            key_update_dir,
            is_transposed,
            stages>(
            mat_key, key_payload, key_prefetch_payload, boundary_x, boundary_y);

        query_payload.template update_tdesc<query_update_dir>(head_size_stride);
        query_acc_tile_t mat_query_acc;
        key_acc_tile_t mat_key_acc;
        subgroup::elemwise_cvt(mat_query_acc, mat_query);
        subgroup::elemwise_cvt(mat_key_acc, mat_key);
        score_sub += mat_vec_mul<accum_t, head_size_stride, key_acc_tile_t, 0>(
            mat_query_acc.reg, mat_key_acc);
      }

      score_sub *= args.sm_scale;
      uint32_t remained_len = ctx.context_len - bid * block_size;
      if (remained_len < block_size) {
        xetla_mask<block_size> mask =
            xetla_vector_gen<uint32_t, block_size>(1, 1) > remained_len;
        score_sub.xetla_merge(neg_infinity, mask);
      }
    }
  }

  // -------------------- // softmax // -------------------- //

  // Compute softmax of score.
  inline void softmax(score_tile_t& mat_score, arguments_t& args) {
    using wg_reduce_max_t =
        group_reduce_t<score_tile_t, wg_size, reduce_op::max, arch_tag>;
    using wg_reduce_sum_t =
        group_reduce_t<score_tile_t, wg_size, reduce_op::sum, arch_tag>;

    wg_reduce_max_t wg_reduce_max(
        ctx.num_blocks_per_sg, ctx.sg_id, 0, slm_offset_softmax);
    accum_t group_max = wg_reduce_max(mat_score);
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive();

    mat_score.reg -= group_max;
    mat_score.reg = xetla_exp<accum_t>(mat_score.reg);

    wg_reduce_sum_t wg_reduce_sum(
        ctx.num_blocks_per_sg, ctx.sg_id, 0, slm_offset_softmax);
    if constexpr (wg_size > 1) {
      ctx.nbarrier.wait();
    }
    accum_t group_sum = wg_reduce_sum(mat_score);
    mat_score.reg /= group_sum;

    if (use_partition && ctx.sg_id == 0) {
      // store the max and exp_sum
      using tile_desc_t = subgroup::tile_desc_t<1, 1, 1, 1>;
      using tile_t = subgroup::tile_t<accum_t, tile_desc_t>;
      using global_st_payload_t = subgroup::mem_payload_t<
          mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
          tile_desc_t,
          msg_type::block_1d,
          arch_tag>;

      uint32_t boundary_y = args.num_seqs * args.num_heads;
      int32_t start_y = ctx.seq_id * args.num_heads + ctx.head_id;
      global_st_payload_t max_logits_st_payload(
          args.max_logits,
          ctx.max_num_partitions,
          boundary_y,
          ctx.max_num_partitions,
          ctx.partition_id,
          start_y);
      tile_t max_logit(group_max);
      subgroup::tile_store(max_logit, max_logits_st_payload);

      global_st_payload_t exp_sums_st_payload(
          args.exp_sums,
          ctx.max_num_partitions,
          boundary_y,
          ctx.max_num_partitions,
          ctx.partition_id,
          start_y);
      tile_t exp_sum(group_sum);
      subgroup::tile_store(exp_sum, exp_sums_st_payload);
    }
  }

  // -------------------- // comput_out // -------------------- //

  using out_tile_desc_t = subgroup::tile_desc_t<max_head_size, 1, 16, 1>;
  using out_tile_t = subgroup::tile_t<accum_t, out_tile_desc_t>;

  // Compute output.
  inline void comput_out(
      score_tile_t& mat_score,
      out_tile_t& mat_out,
      arguments_t& args) {
    constexpr uint32_t sg_tile_size_head =
        std::min(uint32_t(head_size_stride), uint32_t(32 / sizeof(scalar_t)));
    constexpr uint32_t sg_tile_size_block = std::min(uint32_t(block_size), 32u);
    constexpr uint32_t sg_tile_size_x = !has_2d_ld_st
        ? sg_tile_size_block
        : std::min(uint32_t(block_size), uint32_t(32 / sizeof(scalar_t)));
    constexpr uint32_t sg_tile_size_y = !has_2d_ld_st ? sg_tile_size_head : 16;
    constexpr uint32_t tile_size_1d = arch_tag >= gpu_arch::XeHpc
        ? block_size * head_size_stride
        : block_size;

    using value_tile_desc_t = subgroup::tile_desc_t<
        block_size,
        head_size_stride,
        sg_tile_size_x,
        sg_tile_size_y>;
    using value_1d_tile_desc_t =
        subgroup::tile_desc_t<tile_size_1d, 1, sg_tile_size_x, 1>;
    using value_tile_t = subgroup::tile_t<scalar_t, value_tile_desc_t>;
    using value_1d_tile_t = subgroup::tile_t<scalar_t, value_1d_tile_desc_t>;
    using value_acc_tile_t = subgroup::tile_t<accum_t, value_tile_desc_t>;

    using value_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            value_tile_desc_t,
            msg_type::block_2d,
            arch_tag>,
        subgroup::mem_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            value_1d_tile_desc_t,
            msg_type::block_1d,
            arch_tag>>;
    using value_prefetch_payload_t = std::conditional_t<
        has_2d_ld_st,
        subgroup::prefetch_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            value_tile_desc_t,
            1,
            arch_tag>,
        subgroup::prefetch_payload_t<
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
            value_1d_tile_desc_t,
            1,
            arch_tag>>;
    constexpr tdesc_update_dir value_update_dir = tdesc_update_dir::y_dir;

    value_tile_t mat_value;

    // iterate over context blocks
    for (int bid = ctx.sg_id + ctx.start_block_id, row_i = 0;
         bid < ctx.end_block_id;
         bid += wg_size, row_i++) {
      // get the physical block id from block_table
      const int block_id = ctx.block_table[bid];

      // Note: we didn't add correct boundary_x as odd seqlen breaks 2d load
      // assumption. As previous score already set unvalid seqlen to zero, it's
      // ok to load wrong v_cache for unvalid seq
      constexpr uint32_t boundary_x = block_size;
      int32_t start_y = block_id * ctx.kv_block_stride + ctx.kv_head_stride;
      uint32_t boundary_y = start_y + args.head_size;

      value_payload_t value_payload(
          args.value_cache, boundary_x, boundary_y, block_size, 0, start_y);
      value_prefetch_payload_t value_prefetch_payload(
          args.value_cache, boundary_x, boundary_y, block_size, 0, start_y);

#pragma unroll
      for (int i = 0; i < stages; i++) {
        constexpr int prefetch_update_stride =
            has_2d_ld_st ? head_size_stride : 1;
        subgroup::tile_prefetch(value_prefetch_payload);
        value_prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(
            prefetch_update_stride);
      }

      auto score_sub =
          mat_score.reg.xetla_select<block_size, 1>(row_i * block_size);

      for (int i = 0; i < ctx.loop_count; i++) {
        tile_load_2d<
            value_tile_t,
            value_payload_t,
            value_prefetch_payload_t,
            value_update_dir,
            false,
            stages>(
            mat_value,
            value_payload,
            value_prefetch_payload,
            boundary_x,
            boundary_y);

        if constexpr (stages != 0 && has_2d_ld_st) {
          value_prefetch_payload.template update_tdesc<value_update_dir>(
              head_size_stride);
        }
        value_acc_tile_t mat_value_acc;
        subgroup::elemwise_cvt(mat_value_acc, mat_value);
        auto out_sub =
            mat_out.reg.xetla_select<head_size_stride, 1>(i * head_size_stride);
        out_sub += mat_vec_mul<accum_t, block_size, value_acc_tile_t, 1>(
            score_sub, mat_value_acc);
      }
    }
  }

  // -------------------- // collect_out // -------------------- //

  inline void collect_out(out_tile_t& mat_out, arguments_t& args) {
    constexpr uint32_t sg_tile_size_x =
        std::min(uint32_t(head_size_per_sg), uint32_t(32 / sizeof(scalar_t)));
    constexpr uint32_t sg_tile_size_y = std::min(uint32_t(wg_size), 16u);

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
    using global_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        st_tile_desc_t,
        msg_type::block_1d,
        arch_tag>;

    // store out data of each subgroup to slm
    int32_t start_y = ctx.sg_id;
    local_st_payload_t local_st_payload(
        slm_offset_out, max_head_size, wg_size, max_head_size, 0, start_y);
    subgroup::tile_store(mat_out, local_st_payload);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();

    // load out data to register
    uint32_t boundary_x = args.head_size;
    uint32_t boundary_y =
        args.num_seqs * args.num_heads * ctx.max_num_partitions;
    uint32_t pitch = args.head_size;
    int32_t start_x = ctx.sg_id * head_size_per_sg;
    start_y = ctx.seq_id * args.num_heads * ctx.max_num_partitions +
        ctx.head_id * ctx.max_num_partitions + ctx.partition_id;

    if (start_x < args.head_size) {
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
  static inline sycl::nd_range<3> get_nd_range(
      uint32_t num_seqs,
      uint32_t num_heads,
      uint32_t max_num_partitions) {
    static const sycl::range<3> local_range = sycl::range<3>{1, 1, wg_size};
    sycl::range<3> group_range =
        sycl::range<3>{num_heads, num_seqs, max_num_partitions};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  // Entrance of the functor
  inline KERNEL_FUNC void operator()(
      sycl::nd_item<3>& item,
      arguments_t& args) {
    // initialization
    ctx.init(item, args);
    if (use_partition && ctx.start_block_id >= ctx.end_block_id) {
      return;
    }

    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    preload_query(args);

    score_tile_t mat_score(0.0f);
    compute_score(mat_score, args);

    softmax(mat_score, args);

    out_tile_t mat_out(0.0f);
    comput_out(mat_score, mat_out, args);
    collect_out(mat_out, args);
  }
};

// ======================== // Reduce kernel // ======================== //

template <
    typename policy,
    typename scalar_t,
    typename index_t,
    gpu_arch arch_tag>
class paged_attention_reduce {
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
    const int loop_count = DIVIDE_ROUND_UP(args.head_size, head_size_stride);

    int32_t base_x =
        (ctx.seq_id * args.num_heads + ctx.head_id) * args.max_num_partitions;
    uint32_t boundary_x = base_x + ctx.num_partitions;
    for (int start_x = ctx.sg_id * partition_stride, row_i = 0;
         start_x < ctx.num_partitions;
         start_x += ctx.wg_partition_stride, row_i++) {
      tmp_payload_t tmp_payload(
          args.tmp_out,
          args.head_size,
          boundary_x,
          args.head_size,
          0,
          base_x + start_x);
      tmp_prefetch_payload_t tmp_prefetch_payload(
          args.tmp_out,
          args.head_size,
          boundary_x,
          args.head_size,
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
            args.head_size,
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
    using global_st_payload_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        st_tile_desc_t,
        msg_type::block_1d,
        arch_tag>;

    // store out data of each subgroup to slm
    int32_t start_y = ctx.sg_id;
    local_st_payload_t local_st_payload(
        slm_offset_out, max_head_size, wg_size, max_head_size, 0, start_y);
    subgroup::tile_store(mat_out, local_st_payload);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();

    // load out data to register
    uint32_t boundary_x = args.head_size;
    uint32_t boundary_y = args.num_seqs * args.num_heads;
    uint32_t pitch = args.head_size;
    int32_t start_x = ctx.sg_id * head_size_per_sg;
    start_y = ctx.seq_id * args.num_heads + ctx.head_id;

    if (start_x < args.head_size) {
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

#undef DIVIDE_ROUND_UP

} // namespace attention

} // namespace gpu::xetla
