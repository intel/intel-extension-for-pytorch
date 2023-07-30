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

#include "xetla.hpp"

namespace gpu::xetla::fmha {

// --------------------- // imem_desc_t // ---------------------- //

/// @brief Memory struct for tile load and store
/// @tparam dtype Is the data type
/// @tparam lanes Is the number of indexes for one tile
/// @tparam memory_space Is the data location
/// @tparam memory_layout Is the memory layout
template <typename dtype, uint32_t lanes, mem_layout layout, mem_space space>
struct imem_desc_t {};

template <typename dtype_, uint32_t lanes_>
struct imem_desc_t<dtype_, lanes_, mem_layout::row_major, mem_space::global> {
  using dtype = dtype_;
  static constexpr uint32_t lanes = lanes_;
  static constexpr mem_layout layout = mem_layout::row_major;
  static constexpr mem_space space = mem_space::global;

  using desc_t = mem_desc_t<dtype, layout, space>;

  dtype* addr0;
  dtype* addr1;
  uint32_t width;
  int32_t offset;

  xetla_vector<int32_t, lanes> index0;
  xetla_vector<int32_t, lanes> index1;
  uint32_t lanes0;
  uint32_t lanes1;

  inline imem_desc_t() = default;

  inline void init(dtype* address0, dtype* address1, uint32_t surface_width) {
    addr0 = address0;
    addr1 = address1;
    width = surface_width;
  }

  inline void update_index(
      xetla_vector<int32_t, lanes> index0_,
      xetla_vector<int32_t, lanes> index1_,
      uint32_t lanes0_,
      uint32_t lanes1_) {
    index0 = index0_;
    index1 = index1_;
    lanes0 = lanes0_;
    lanes1 = lanes1_;
  }

  inline void set_offset(int32_t offset_) {
    offset = offset_;
  }

  inline bool init_tdesc(uint32_t lane, desc_t& desc) {
    if (lane < lanes0) {
      int32_t idx = index0[lane];
      uint32_t height = idx + 1;
      desc.init(addr0, {width, height, width}, {offset, idx});
    } else {
      lane -= lanes0;
      if (lane < lanes1) {
        int32_t idx = index1[lane];
        uint32_t height = idx + 1;
        desc.init(addr1, {width, height, width}, {offset, idx});
      } else {
        return false;
      }
    }
    return true;
  }
};

// --------------------- // iload_tile // ---------------------- //

template <typename imem_desc_t>
struct load_type {
  static constexpr bool is_global_row =
      ((imem_desc_t::space == mem_space::global) &&
       (imem_desc_t::layout == mem_layout::row_major));
};

template <typename tile_t, typename imem_desc_t>
inline typename std::enable_if_t<load_type<imem_desc_t>::is_global_row>
iload_tile(tile_t& tile, imem_desc_t& imem_desc) {
  using dtype = typename imem_desc_t::dtype;
  static constexpr uint32_t lanes = imem_desc_t::lanes;
  static constexpr mem_layout layout = imem_desc_t::layout;
  static constexpr mem_space space = imem_desc_t::space;

  using dtype_acc = typename tile_t::dtype;
  using tile_desc = typename tile_t::tile_desc;
  static constexpr uint32_t width = tile_desc::tile_size_x;
  static constexpr uint32_t height = tile_desc::tile_size_y;

  static_assert(lanes == height, "the tile height and lanes don't match");
  static_assert(width >= 16, "width is expected to be larger than 16");
  using lane_tile_desc_t = subgroup::tile_desc_t<width, 1, 16, 1>;
  using lane_tile_t = subgroup::tile_t<dtype, lane_tile_desc_t>;
  using lane_payload_t = subgroup::
      mem_payload_t<dtype, lane_tile_desc_t, msg_type::block_1d, layout, space>;
  using lane_mem_desc_t = mem_desc_t<dtype, layout, space>;

  lane_tile_t lane_tile;
  lane_mem_desc_t lane_mem_desc;
  lane_payload_t lane_payload;

#pragma unroll
  for (int i = 0; i < lanes; i++) {
    if (imem_desc.init_tdesc(i, lane_mem_desc)) {
      lane_payload.init(lane_mem_desc);
      subgroup::tile_load(lane_tile, lane_payload);
    } else {
      lane_tile.init(0);
    }

    tile.reg.xetla_select<width, 1>(i * width) =
        xetla_cvt<dtype_acc, dtype, width>(lane_tile.reg);
  }
}

template <typename tile_t, typename imem_desc_t>
inline typename std::enable_if_t<load_type<imem_desc_t>::is_global_row>
iload_tile(tile_t& tile, imem_desc_t& imem_desc, int lane_idx) {
  using dtype = typename imem_desc_t::dtype;
  static constexpr uint32_t lanes = imem_desc_t::lanes;
  static constexpr mem_layout layout = imem_desc_t::layout;
  static constexpr mem_space space = imem_desc_t::space;

  using dtype_acc = typename tile_t::dtype;
  using tile_desc = typename tile_t::tile_desc;
  static constexpr uint32_t width = tile_desc::tile_size_x;

  static_assert(width >= 16, "width is expected to be larger than 16");
  using lane_tile_desc_t = subgroup::tile_desc_t<width, 1, 16, 1>;
  using lane_tile_t = subgroup::tile_t<dtype, lane_tile_desc_t>;
  using lane_payload_t = subgroup::
      mem_payload_t<dtype, lane_tile_desc_t, msg_type::block_1d, layout, space>;
  using lane_mem_desc_t = mem_desc_t<dtype, layout, space>;

  lane_tile_t lane_tile;
  lane_mem_desc_t lane_mem_desc;
  lane_payload_t lane_payload;

  if (imem_desc.init_tdesc(lane_idx, lane_mem_desc)) {
    lane_payload.init(lane_mem_desc);
    subgroup::tile_load(lane_tile, lane_payload);
  } else {
    lane_tile.init(0);
  }

  tile.reg = xetla_cvt<dtype_acc, dtype, width>(lane_tile.reg);
}

template <typename tile_t, typename imem_desc_t>
inline typename std::enable_if_t<load_type<imem_desc_t>::is_global_row>
iprefetch_tile(imem_desc_t& imem_desc) {
  using dtype = typename imem_desc_t::dtype;
  static constexpr uint32_t lanes = imem_desc_t::lanes;
  static constexpr mem_layout layout = imem_desc_t::layout;
  static constexpr mem_space space = imem_desc_t::space;

  using dtype_acc = typename tile_t::dtype;
  using tile_desc = typename tile_t::tile_desc;
  static constexpr uint32_t width = tile_desc::tile_size_x;
  static constexpr uint32_t height = tile_desc::tile_size_y;

  static_assert(lanes == height, "the tile height and lanes don't match");

  using lane_tile_desc_t = subgroup::tile_desc_t<width, 1, width, 1>;
  using lane_tile_t = subgroup::tile_t<dtype, lane_tile_desc_t>;
  using prefetch_payload_t = subgroup::prefetch_payload_t<
      dtype,
      lane_tile_desc_t,
      layout,
      space,
      1,
      gpu_arch::Xe>;
  using lane_mem_desc_t = mem_desc_t<dtype, layout, space>;

  lane_mem_desc_t lane_mem_desc;
#pragma unroll
  for (int i = 0; i < lanes; i++) {
    if (imem_desc.init_tdesc(i, lane_mem_desc)) {
      prefetch_payload_t prefetch_payload(lane_mem_desc);
      subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          prefetch_payload);
    }
  }
}

// --------------------- // group_1d_reduce_t // ---------------------- //

template <typename mat_t, uint32_t kNumSg, reduce_op reduce_kind>
struct group_1d_reduce_t {
  using T = typename mat_t::dtype;
  static constexpr uint32_t tile_size_x = mat_t::tile_desc::tile_size_x;
  static_assert(
      mat_t::tile_desc::tile_size_y == 1,
      "the tile_size_y should be 1d");

  // store results of subgroup to slm
  using store_tile_desc = subgroup::tile_desc_t<1, 1, 1, 1>;
  using store_tile_t = subgroup::tile_t<T, store_tile_desc>;
  using store_payload_t = subgroup::mem_payload_t<
      T,
      store_tile_desc,
      msg_type::block_1d,
      mem_layout::row_major,
      mem_space::local>;
  // load all subgroup results together
  using load_tile_desc = subgroup::tile_desc_t<kNumSg, 1, kNumSg, 1>;
  using load_tile_t = subgroup::tile_t<T, load_tile_desc>;
  using load_payload_t = subgroup::mem_payload_t<
      T,
      load_tile_desc,
      msg_type::block_1d,
      mem_layout::row_major,
      mem_space::local>;

  xetla_nbarrier_t<kNumSg, kNumSg> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  inline group_1d_reduce_t() = default;
  inline group_1d_reduce_t(
      uint32_t sg_id_,
      uint32_t nbarrier_id,
      uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
  }

  inline KERNEL_FUNC T operator()(mat_t& src) {
    T ret = xetla_reduce<T, T, tile_size_x, reduce_kind>(src.reg);
    if constexpr (kNumSg == 1)
      return ret;

    store_tile_t sg_store;
    store_payload_t sg_store_payload(slm_base, kNumSg, 1, kNumSg, sg_id, 0);
    sg_store.reg = ret;
    subgroup::tile_store(sg_store, sg_store_payload);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    load_tile_t sg_load;
    load_payload_t sg_load_payload(slm_base, kNumSg, 1, kNumSg, 0, 0);
    subgroup::tile_load(sg_load, sg_load_payload);

    ret = xetla_reduce<T, T, kNumSg, reduce_kind>(sg_load.reg);
    return ret;
  }
};

} // namespace gpu::xetla::fmha
