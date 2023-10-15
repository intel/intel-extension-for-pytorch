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

#pragma once

#include "prefetch_xe.h"
#include "xetla.hpp"

namespace gpu::xetla::fmha {

/// @brief Memory struct for indexed tile load
template <typename dtype_, uint32_t N_, uint32_t Beams_, uint32_t LoadSize_>
struct imem_desc_t {
  using dtype = dtype_;
  static constexpr uint32_t N = N_;
  static constexpr uint32_t Beams = Beams_;
  static constexpr uint32_t LoadSize = LoadSize_;
  static constexpr uint32_t SZ = N * Beams;

  static constexpr mem_layout layout = mem_layout::row_major;
  static constexpr mem_space space = mem_space::global;

  dtype* addr_;
  uint32_t width_;
  uint32_t pitch_;

  xetla_vector<int32_t, SZ> index_;
  uint32_t total_;

  int32_t offset_;
  uint32_t lane_;
  uint32_t beam_;
  int32_t offset_pre_;
  uint32_t lane_pre_;
  uint32_t beam_pre_;

  inline imem_desc_t() = default;
  inline imem_desc_t(dtype* address, uint32_t width, uint32_t pitch) {
    addr_ = address;
    width_ = width;
    pitch_ = pitch;
  }

  inline void init(dtype* address, uint32_t width, uint32_t pitch) {
    addr_ = address;
    width_ = width;
    pitch_ = pitch;
  }

  inline void init_index(xetla_vector<int32_t, SZ> index, uint32_t total) {
    index_ = index;
    total_ = total;
    offset_ = 0;
    lane_ = 0;
    beam_ = 0;
    offset_pre_ = 0;
    lane_pre_ = 0;
    beam_pre_ = 0;
  }

  inline void update_tdesc() {
    if (++lane_ >= N_) {
      lane_ = 0;
      if (++beam_ >= Beams) {
        beam_ = 0;
        offset_ += LoadSize;
      }
    }
  }

  inline void update_prefetch_tdesc() {
    if (++lane_pre_ >= N_) {
      lane_pre_ = 0;
      if (++beam_pre_ >= Beams) {
        beam_pre_ = 0;
        offset_pre_ += LoadSize;
      }
    }
  }

  template <typename tile_t>
  inline void iload_tile(tile_t& tile) {
    using dtype_acc = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    static_assert(
        tile_desc::tile_size_x == LoadSize,
        "tile_size_x should equal LoadSize");
    static_assert(tile_desc::tile_size_y == 1, "the tile should be 1d");

    using lane_tile_desc_t = subgroup::tile_desc_t<LoadSize, 1, 16, 1>;
    using lane_tile_t = subgroup::tile_t<dtype, lane_tile_desc_t>;
    using lane_payload_t = subgroup::mem_payload_t<
        dtype,
        lane_tile_desc_t,
        msg_type::block_1d,
        layout,
        space>;

    lane_tile_t lane_tile;

    if (offset_ < width_ && lane_ < total_) {
      int32_t idx = index_[lane_ * Beams + beam_];
      lane_payload_t lane_payload(addr_, width_, idx + 1, pitch_, offset_, idx);
      subgroup::tile_load(lane_tile, lane_payload);
    } else {
      lane_tile.init(0);
    }

    tile.reg = xetla_cvt<dtype_acc, dtype, LoadSize>(lane_tile.reg);
  }

  inline void iprefetch_tile() {
    using lane_tile_desc_t = subgroup::tile_desc_t<LoadSize, 1, 16, 1>;
    using lane_payload_t = subgroup::ext::
        prefetch_payload_t<dtype, lane_tile_desc_t, layout, space, 1>;

    if (offset_pre_ < width_ && lane_pre_ < total_) {
      int32_t idx = index_[lane_pre_ * Beams + beam_pre_];
      lane_payload_t lane_payload(
          addr_, width_, idx + 1, pitch_, offset_pre_, idx);
      subgroup::ext::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          lane_payload);
    }
  }
};

template <typename T, uint32_t size, uint32_t reduce_size>
inline xetla_vector<T, reduce_size> partial_reduce(xetla_vector<T, size> src) {
  static_assert(
      size % reduce_size == 0, "size should be a multiple of reduce_size");
  xetla_vector<T, reduce_size> ret(0);

#pragma unroll
  for (int i = 0; i < size / reduce_size; i++) {
    auto src_sub = src.xetla_select<reduce_size, 1>(i * reduce_size);
    ret += src_sub;
  }

  return ret;
}

// ==================== // tile_mask_t // ================== //

template <typename mat_t>
struct tile_mask_t {
  using accum_t = typename mat_t::dtype;
  static constexpr accum_t kNegInfinity = INFINITY * -1;
  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  static constexpr uint32_t block_size_x = mat_t::block_size_x;
  static constexpr uint32_t block_size_y = mat_t::block_size_y;
  static constexpr int32_t num_block_x = mat_t::num_block_x;
  static constexpr uint32_t block_elems = mat_t::block_elems;

  inline static void padding_mask(mat_t& src, int num_keep) {
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(0, num_keep - start_x);

        if (num_keep_blk < block_size_x) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src.reg
                  .xetla_select<block_elems, 1>(
                      (i * num_block_x + j) * block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < block_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(num_keep - start_x, 0);

        if (num_keep_blk < block_size_x) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src.reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }
};

// ==================== // group_row_reduce_t // ================== //

template <typename mat_t, uint32_t kNumSg, reduce_op reduce_kind>
struct group_row_reduce_t {
  using T = typename mat_t::dtype;
  static constexpr uint32_t kNum = mat_t::tile_desc::tile_size_y;
  static constexpr uint32_t kTotal = kNum * kNumSg;

  // store results of subgroup to slm
  using store_tile_desc =
      subgroup::tile_desc_t<kNum, 1, kNum, 1, reg_layout::tiled>;
  using store_tile_t = subgroup::tile_t<T, store_tile_desc>;
  using store_payload_t = subgroup::mem_payload_t<
      T,
      store_tile_desc,
      msg_type::block_1d,
      mem_layout::row_major,
      mem_space::local,
      gpu_arch::Xe>;
  // load all subgroup results together
  using load_tile_desc =
      subgroup::tile_desc_t<kTotal, 1, kTotal, 1, reg_layout::tiled>;
  using load_tile_t = subgroup::tile_t<T, load_tile_desc>;
  using load_payload_t = subgroup::mem_payload_t<
      T,
      load_tile_desc,
      subgroup::msg_type_v<load_tile_desc, mem_space::local>,
      mem_layout::row_major,
      mem_space::local,
      gpu_arch::Xe>;

  xetla_nbarrier_t<kNumSg, kNumSg> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  inline group_row_reduce_t() = default;
  inline group_row_reduce_t(
      uint32_t sg_id_,
      uint32_t nbarrier_id,
      uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
  }

  inline KERNEL_FUNC xetla_vector<T, kNum> operator()(mat_t& src) {
    xetla_vector<T, kNum> ret =
        subgroup::tile_reduce<reduce_kind, T, T, 1>(src);
    if constexpr (kNumSg == 1)
      return ret;

    store_tile_t sg_store;
    store_payload_t sg_store_payload(
        slm_base, kTotal, 1, kTotal, sg_id * kNum, 0);
    sg_store.reg = ret;
    subgroup::tile_store(sg_store, sg_store_payload);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    load_tile_t sg_load;
    load_payload_t sg_load_payload(slm_base, kTotal, 1, kTotal, 0, 0);
    subgroup::tile_load(sg_load, sg_load_payload);

    ret = recur_row_reduce<reduce_kind, T, kNum, kNumSg>(sg_load.reg);
    // auto data_2d = sg_load.reg.xetla_format<T, kNumSg, kNum>();
    //     ret = data_2d.row(0);
    // #pragma unroll
    //     for (int i = 1; i < kNumSg; i++) {
    //       ret = reduce_helper<reduce_kind, T, kNum>(data_2d.row(i), ret);
    //     }
    return ret;
  }
};

} // namespace gpu::xetla::fmha
