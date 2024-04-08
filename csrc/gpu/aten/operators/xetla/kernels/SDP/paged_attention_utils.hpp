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

#include <type_traits>
#include "common/core/common.hpp"
#include "subgroup/tile/api.hpp"
#include "xetla.hpp"

namespace gpu::xetla {

namespace attention {
constexpr float neg_infinity = INFINITY * -1;

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

template <typename mat_t, uint32_t wg_size, reduce_op reduce_kind>
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
      gpu_arch::XeHpc>;
  // load all subgroup results together
  using local_ld_tile_desc =
      subgroup::tile_desc_t<wg_size, 1, wg_size, 1, reg_layout::tiled>;
  using local_ld_tile_t = subgroup::tile_t<dtype, local_ld_tile_desc>;
  using local_ld_payload_t = subgroup::mem_payload_t<
      mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
      local_ld_tile_desc,
      msg_type::block_1d,
      gpu_arch::XeHpc>;
  // local variables
  xetla_nbarrier_t<wg_size, wg_size> nbarrier;
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
