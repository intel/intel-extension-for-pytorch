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

#include <cmath>
#include <type_traits>
#include "common/core/common.hpp"
#include "common/core/common_types.hpp"
#include "common/core/memory.hpp"
#include "common/utils/misc.hpp"
#include "subgroup/tile/api.hpp"
#include "xetla.hpp"

namespace gpu::xetla {

namespace attention {
constexpr float neg_infinity = INFINITY * -1;

inline void SW_BARRIER() {
#if __INTEL_LLVM_COMPILER >= 20250000
#if defined(__SYCL_DEVICE_ONLY__)
  __asm__ volatile("fence_sw" : : :);
#endif // __SYCL_DEVICE_ONLY__
#else
  __ESIMD_NS::fence<__ESIMD_NS::fence_mask::sw_barrier>();
#endif // __INTEL_LLVM_COMPILER >= 20250000
}

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

// Matrix-Vector Multiplication, which computes the matrix vector product.
template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 1 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_x == N,
    xetla_vector<dtype, mat_t::tile_size_y>>
mat_vec_mul(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;
  // The calculation process:
  // 1) allocate a temp buffer
  // 2) compute the product of m x v by each block
  // 3) accumulate the result into temp buffer
  // 4) reduce within the temp buffer
  xetla_vector<dtype, tile_size_y * block_size_x> acc;
#pragma unroll
  for (int i = 0; i < num_block_y; i++) {
    // j = 0
    {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(0);
      auto mat_sub =
          mat.reg.xetla_select<block_elems, 1>(i * num_block_x * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < block_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (i * block_size_y + row_i) * block_size_x);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
    // j = 1...
#pragma unroll
    for (int j = 1; j < num_block_x; j++) {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(j * block_size_x);
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((i * num_block_x + j) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < block_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (i * block_size_y + row_i) * block_size_x);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  // process the tail
  if constexpr (remained_size_y != 0) {
    constexpr uint32_t remained_start_y = num_block_y * block_size_y;
    constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
    // j = 0
    {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(0);
      auto mat_sub = mat.reg
                         .xetla_select<remained_block_elems, 1>(
                             remained_start_y * tile_size_x)
                         .xetla_format<dtype, remained_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < remained_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (remained_start_y + row_i) * block_size_x);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
    // j = 1...
#pragma unroll
    for (int j = 1; j < num_block_x; j++) {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(j * block_size_x);
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + j * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < block_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (remained_start_y + row_i) * block_size_x);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  return recur_col_reduce<reduce_op::sum, dtype, block_size_x, tile_size_y>(
      acc);
}
template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x>>
mat_vec_mul(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  using tile_desc_t = subgroup::
      tile_desc_t<tile_size_x, block_size_y, block_size_x, block_size_y>;
  using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  tile_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);

  return res;
}

template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x * mat_t::tile_size_y>>
mat_vec_mul_broadcast(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  // using tile_desc_t = subgroup::
  //     tile_desc_t<tile_size_x, tile_size_y, block_size_x, block_size_y>;
  // using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  mat_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            (j * num_block_x + i) * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            remained_start_y * tile_size_x + i * remained_block_elems +
            row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  // auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);
  return acc.reg;
}

template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x * mat_t::tile_size_y>>
mat_vec_div_broadcast(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  using tile_desc_t = subgroup::
      tile_desc_t<tile_size_x, tile_size_y, block_size_x, block_size_y>;
  using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  tile_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      vec_sub = 1.0f / vec_sub;
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x +
            j * num_block_x * block_elems);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        vec_sub = 1.0f / vec_sub;
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x +
            remained_start_y * tile_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        vec_sub = 1.0f / vec_sub;
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  // auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);
  return acc.reg;
}

template <
    typename mat_t,
    uint32_t wg_size,
    reduce_op reduce_kind,
    gpu_arch arch_tag>
struct group_reduce_t {
  using dtype = typename mat_t::dtype;
  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;

  // store results of subgroup to slm
  using local_st_tile_desc =
      subgroup::tile_desc_t<1, 1, 1, 1, reg_layout::tiled>;
  using local_st_tile_t = subgroup::tile_t<dtype, local_st_tile_desc>;
  using local_st_payload_t = subgroup::mem_payload_t<
      mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
      local_st_tile_desc,
      msg_type::block_1d,
      arch_tag>;
  // load all subgroup results together
  using local_ld_tile_desc =
      subgroup::tile_desc_t<wg_size, 1, wg_size, 1, reg_layout::tiled>;
  using local_ld_tile_t = subgroup::tile_t<dtype, local_ld_tile_desc>;
  using local_ld_payload_t = subgroup::mem_payload_t<
      mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
      local_ld_tile_desc,
      msg_type::block_1d,
      arch_tag>;
  // local variables
  xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  uint32_t num_rows;
  inline group_reduce_t() = default;
  inline group_reduce_t(
      uint32_t num_rows_,
      uint32_t sg_id_,
      uint32_t nbarrier_id,
      uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
    num_rows = num_rows_;
  }

  inline KERNEL_FUNC dtype operator()(mat_t& src) {
    // local reduction
    xetla_vector<dtype, 1> local_res(0);
    if constexpr (reduce_kind == reduce_op::max) {
      local_res[0] = neg_infinity;
    }

    if (num_rows > 0) {
      xetla_vector<dtype, tile_size_x> res =
          src.reg.xetla_select<tile_size_x, 1>(0);
      for (int row_i = 1; row_i < num_rows; row_i++) {
        res = reduce_helper<reduce_kind, dtype, tile_size_x>(
            res, src.reg.xetla_select<tile_size_x, 1>(row_i * tile_size_x));
      }
      local_res = recur_col_reduce<reduce_kind, dtype, tile_size_x, 1>(res);
    }

    if constexpr (wg_size == 1)
      return local_res[0];

    local_st_tile_t local_st;
    local_st_payload_t local_st_payload(
        slm_base, wg_size, 1, wg_size, sg_id, 0);
    local_st.reg = local_res;
    subgroup::tile_store(local_st, local_st_payload);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    local_ld_tile_t local_ld;
    local_ld_payload_t local_ld_payload(slm_base, wg_size, 1, wg_size, 0, 0);
    subgroup::tile_load(local_ld, local_ld_payload);

    return xetla_reduce<dtype, dtype, wg_size, reduce_kind>(local_ld.reg);
  }
};

} // namespace attention

} // namespace gpu::xetla
