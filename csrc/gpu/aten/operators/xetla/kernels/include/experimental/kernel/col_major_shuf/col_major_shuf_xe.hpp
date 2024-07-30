/*******************************************************************************
 * Copyright (c) 2023-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @file
/// C++ API

#pragma once

#include <experimental/kernel/col_major_shuf/api.hpp>
#include <experimental/kernel/col_major_shuf/common.hpp>
#include <experimental/kernel/col_major_shuf/config.hpp>

namespace gpu::xetla::kernel {
template <
    typename dtype_in_,
    typename dtype_out_,
    typename dtype_gidx_,
    mem_layout mem_layout_in_,
    typename col_major_shuf_attr_,
    gpu_arch arch_>
struct col_major_shuf_t<
    dtype_in_,
    dtype_out_,
    dtype_gidx_,
    mem_layout_in_,
    col_major_shuf_attr_,
    arch_> {
  using dtype_in = dtype_in_;
  using dtype_out = dtype_out_;
  using dtype_gidx = dtype_gidx_;
  using col_major_shuf_attr = col_major_shuf_attr_;

  static constexpr mem_layout mem_layout_in = mem_layout_in_;

  static_assert(
      std::is_same<dtype_in, dtype_out>::value,
      "only support in/out data type must be same now.");
  static_assert(
      mem_layout_in == mem_layout::row_major,
      "only support row_major input now.");
  static_assert(
      std::is_same<dtype_gidx, uint32_t>::value,
      "dtype_gidx must be uint32_t");

  static constexpr uint32_t wg_tile_x = col_major_shuf_attr::wg_tile_x;
  static constexpr uint32_t wg_tile_y = col_major_shuf_attr::wg_tile_y;
  static constexpr uint32_t sg_tile_x = col_major_shuf_attr::sg_tile_x;
  static constexpr uint32_t sg_tile_y = col_major_shuf_attr::sg_tile_y;

  static constexpr uint32_t tile_size_x = sg_tile_x;
  static constexpr uint32_t tile_size_y = sg_tile_y;

  static constexpr uint32_t block_size_x =
      col_major_shuf_attr::load_block_size; // TODO(zhe:) add load block size
                                            // check under different arch

  static constexpr uint32_t dev_mem_align = 64;
  using mem_desc_store_tile_t = mem_desc_t<
      dtype_in,
      mem_layout_in,
      mem_space::global,
      dev_mem_align / sizeof(dtype_in)>;
  using store_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x,
      tile_size_y,
      block_size_x,
      tile_size_y,
      reg_layout::tiled>;
  using store_tile_t = subgroup::tile_t<dtype_out, store_tile_desc_t>;
  using store_tile_payload_t = subgroup::mem_payload_t<
      mem_desc_store_tile_t,
      store_tile_desc_t,
      subgroup::msg_type_v<store_tile_desc_t, mem_desc_store_tile_t>,
      arch_>;

  using mem_desc_gidx_t = mem_desc_t<
      dtype_gidx,
      mem_layout::row_major,
      mem_space::global,
      dev_mem_align / sizeof(dtype_gidx)>;
  using gidx_tile_desc_t =
      subgroup::tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
  using gidx_t = subgroup::tile_t<dtype_gidx, gidx_tile_desc_t>;
  using gidx_payload_t = subgroup::mem_payload_t<
      mem_desc_gidx_t,
      gidx_tile_desc_t,
      subgroup::msg_type_v<gidx_tile_desc_t, mem_desc_gidx_t>,
      arch_>;

  struct arguments_t {
    dtype_in* mat_in_ptr;
    dtype_out* mat_out_ptr;
    dtype_gidx* gidx_ptr;
    uint32_t matrix_x;
    uint32_t matrix_y;
  };

  __XETLA_API static void call(sycl::nd_item<3>& item, arguments_t& args) {
    int gid_x = item.get_group(2);
    int gid_y = item.get_group(1);
    int x_dim_offset = gid_x * wg_tile_x;
    int y_dim_offset = gid_y * wg_tile_y;
    int tid_x = item.get_local_id(2);
    int tid_y = item.get_local_id(1);
    x_dim_offset += tid_x * sg_tile_x;
    y_dim_offset += tid_y * sg_tile_y;
    mem_desc_gidx_t gidx_desc(
        args.gidx_ptr, {args.matrix_x, 1, args.matrix_x}, {x_dim_offset, 0});
    mem_desc_store_tile_t store_tile_desc(
        args.mat_out_ptr,
        {args.matrix_x, args.matrix_y, args.matrix_x},
        {x_dim_offset, y_dim_offset});

    static constexpr int block_x_num = tile_size_x / block_size_x;
    static constexpr int elt_per_block = block_size_x * tile_size_y;
    store_tile_t store_tile;
    store_tile_payload_t store_tile_payload(store_tile_desc);
    gidx_payload_t gidx_payload(gidx_desc);

#pragma unroll
    for (int block_x = 0; block_x < block_x_num; block_x++) {
      auto gidx = xetla_load_global<
          uint32_t,
          block_size_x,
          cache_hint::cached,
          cache_hint::cached>(
          args.gidx_ptr, gidx_payload.base_offset + block_x * block_size_x);
#pragma unroll
      for (uint32_t row = 0; row < tile_size_y; row++) {
        store_tile.reg.xetla_select<block_size_x, 1>(
            block_x * elt_per_block + row * block_size_x) =
            xetla_load_global<
                dtype_in,
                block_size_x,
                1,
                cache_hint::cached,
                cache_hint::cached>(
                args.mat_in_ptr + (y_dim_offset + row) * args.matrix_x,
                gidx,
                1);
      }
    }
    tile_store(store_tile, store_tile_payload);
  };
};
} // namespace gpu::xetla::kernel
