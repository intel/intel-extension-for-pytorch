/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
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

#include <subgroup/tile/api.hpp>

namespace gpu::xetla::subgroup {

/// @brief Is the tile mma operation functor, specialized for Xe and fpu engine.
template <
    typename matAcc_t_,
    typename matC_t_,
    typename matB_t_,
    typename matA_t_,
    gpu_arch arch_tag_>
struct tile_fma_t {
  using matA_t = matA_t_;
  using matB_t = matB_t_;
  using matC_t = matC_t_;
  using matAcc_t = matAcc_t_;
  using dtype_a = typename matA_t::dtype;
  using dtype_b = typename matB_t::dtype;
  using dtype_acc = typename matAcc_t_::dtype;

  using register_attr =
      typename arch_attr_t<arch_tag_>::template register_attr<>;

  static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
  static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
  static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
  static constexpr uint32_t a_block_size_y = matA_t::block_size_y;
  static constexpr uint32_t a_block_size_x = matA_t::block_size_x;
  static constexpr uint32_t a_block_elems = matA_t::block_elems;

  static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
  static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
  static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
  static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
  static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
  static constexpr uint32_t b_block_elems = matB_t::block_elems;

  static constexpr uint32_t tile_size_m = a_tile_size_y;
  static constexpr uint32_t tile_size_k = a_tile_size_x;
  static constexpr uint32_t tile_size_n = b_tile_size_x;
  static constexpr uint32_t block_size_m = a_block_size_y;
  static constexpr uint32_t block_size_k = a_block_size_x;
  static constexpr uint32_t block_size_n = b_block_size_x;

  static_assert(
      a_tile_size_x == b_tile_size_y,
      "matA tile k should match with matB tile k");
  static_assert(
      a_block_size_x == b_block_size_y,
      "matA block k should match with matB block k");
  static_assert(
      b_block_size_y == matAcc_t::block_size_x,
      "matA block k should match with matAcc block k");
  static_assert(
      a_block_size_y == matAcc_t::block_size_y,
      "mata block m should match with matAcc block m");

  static_assert(tile_size_n == 1, "matB tile n must be 1");
  static_assert(b_block_size_x == 1, "matB block n must be 1");
  __XETLA_API static void mma(
      matAcc_t& acc_dst,
      matAcc_t& acc_src,
      matC_t& c,
      matB_t& b,
      matA_t& a,
      bool reduce) {
#pragma unroll
    for (uint32_t k = 0; k < tile_size_k / block_size_k; k++) {
#pragma unroll
      for (uint32_t m = 0; m < tile_size_m / block_size_m; m++) {
        uint32_t a_block_idx = m * tile_size_k / block_size_k + k;
        auto a_block = a.reg.xetla_select<block_size_m * block_size_k, 1>(
            a_block_idx * matA_t::block_elems);
#pragma unroll
        for (uint32_t n = 0; n < tile_size_n / block_size_n; n++) {
          uint32_t b_block_idx = n * tile_size_k / block_size_k + k;
          auto b_block = b.reg.xetla_select<matB_t::block_elems, 1>(
              b_block_idx * matB_t::block_elems);

          auto src_block =
              acc_src.reg.xetla_select<block_size_m * block_size_k, 1>(
                  m * matAcc_t::block_elems);
          auto dst_block =
              acc_dst.reg.xetla_select<block_size_m * block_size_k, 1>(
                  m * matAcc_t::block_elems);
          fma_core<block_size_m, block_size_n, block_size_k>(
              dst_block, src_block, b_block, a_block);
        }
      }
    }
    if (reduce) {
      reduce_acc_k(acc_dst, c);
    }
  }
  template <int blk_m, int blk_n, int blk_k>
  __XETLA_API static void fma_core(
      xetla_vector_ref<dtype_acc, blk_m * blk_k> __REF__ dst_block,
      xetla_vector_ref<dtype_acc, blk_m * blk_k> __REF__ src_block,
      xetla_vector_ref<dtype_b, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_m * blk_k> __REF__ a_block) {
    static_assert(blk_n == 1, "block n must be 1");
    auto dst_blk_2d = dst_block.xetla_format<dtype_acc, blk_m, blk_k>();
    auto src_blk_2d = src_block.xetla_format<dtype_acc, blk_m, blk_k>();
    auto b_blk_2d = b_block.xetla_format<dtype_b, blk_n, blk_k>();
    auto a_blk_2d = a_block.xetla_format<dtype_a, blk_m, blk_k>();

#pragma unroll
    for (uint32_t m = 0; m < blk_m; m++) {
      auto a_row = a_blk_2d.row(m);
#pragma unroll
      for (uint32_t n = 0; n < blk_n; n++) {
        auto b_row = b_blk_2d.row(n);
        dst_blk_2d.row(m) = b_row * a_row + src_blk_2d.row(m);
      }
    }
  }
  __XETLA_API static void reduce_acc_k(matAcc_t& matAcc, matC_t& matC) {
    // matC  [tx,ty,bx,by](matmul): tile_n,       1, block_n,       1
    // matAcc[tx,ty,bx,by](matmul): tile_n, block_k, block_n, block_k
    // matAcc[tx,ty,bx,by](memory): block_k, tile_n, block_k, block_n

    // static_assert(
    //     matC_t::tile_size_y == 1 && matC_t::block_size_y == 1,
    //     "matDst_t_ tile m and block m should match be 1");
    static_assert(
        matAcc_t::tile_size_y == matC_t::tile_size_y,
        "matAcc_t tile m should match with matDst_t_ tile m");
    static_assert(
        matAcc_t::block_size_y == matC_t::block_size_y,
        "matAcc_t block m should match with matDst_t_ block m");
    static constexpr uint32_t block_k = matAcc_t::block_size_x;
    static constexpr uint32_t block_m = matAcc_t::block_size_y;
    using dtype = matAcc_t::dtype;

#pragma unroll
    for (uint32_t m = 0; m < a_tile_size_y / a_block_size_y; m++) {
      matC.reg.xetla_select<block_m, 1>(m * block_m) =
          recur_col_reduce<reduce_op::sum, dtype, block_k, block_m>(
              matAcc.reg.xetla_select<block_m * block_k, 1>(
                  m * block_m * block_k));
      // matC.reg =
      //     recur_col_reduce<reduce_op::sum, dtype, block_k,
      //     block_m>(matAcc.reg);
    }
  }
};

/// @brief Is the tile mma operation functor, specialized for Xe and fpu
/// engine.
template <
    typename matAcc_dst_t_,
    typename matAcc_src_t_,
    typename matB_t_,
    typename matA_t_,
    gpu_arch arch_tag_>
struct tile_mma_t<
    matAcc_dst_t_,
    matAcc_src_t_,
    matB_t_,
    matA_t_,
    mma_engine::fpu,
    arch_tag_,
    std::enable_if_t<arch_has_fpu<arch_tag_>>> {
  using matA_t = matA_t_;
  using matB_t = matB_t_;
  using matSrc_t = matAcc_src_t_;
  using matDst_t = matAcc_dst_t_;
  using dtype_a = typename matA_t::dtype;
  using dtype_b = typename matB_t::dtype;
  using dtype_src = typename matSrc_t::dtype;
  using dtype_dst = typename matDst_t::dtype;

  using register_attr =
      typename arch_attr_t<arch_tag_>::template register_attr<>;

  static_assert(
      matA_t::reg_transpose,
      "For FMAOp GEMM, the register layout of matA should be col-major");
  static_assert(
      !matB_t::reg_transpose,
      "For FMAOp GEMM, the register layout of matB should be row-major");

  static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
  static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
  static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
  static constexpr uint32_t a_block_size_y = matA_t::block_size_y;
  static constexpr uint32_t a_block_size_x = matA_t::block_size_x;
  static constexpr uint32_t a_block_elems = matA_t::block_elems;

  static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
  static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
  static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
  static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
  static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
  static constexpr uint32_t b_block_elems = matB_t::block_elems;

  static constexpr uint32_t tile_size_m = matDst_t::tile_size_y;
  static constexpr uint32_t tile_size_k = a_tile_size_x;
  static constexpr uint32_t tile_size_n = matDst_t::tile_size_x;
  static constexpr uint32_t tile_elems = tile_size_m * tile_size_n;
  static constexpr uint32_t block_size_n = matDst_t::block_size_x;
  static constexpr uint32_t block_size_k = a_block_size_x;
  static constexpr uint32_t block_size_m = matDst_t::block_size_y;
  static constexpr uint32_t block_elems = block_size_m * block_size_n;

  static_assert(
      tile_size_m == matA_t::tile_size_y,
      "matAcc tile m should match with matA tile m");
  static_assert(
      a_tile_size_x == b_tile_size_y,
      "matA tile k should match with matB tile k");
  static_assert(
      tile_size_n == matB_t::tile_size_x,
      "matAcc tile n should match with matB tile n");
  static_assert(
      block_size_m == a_block_size_y,
      "matAcc block m should match with matA block m");
  static_assert(
      block_size_n == b_block_size_x,
      "matAcc block n should match with matB block n");
  static_assert(
      (tile_size_k % block_size_k) == 0,
      "matAcc tile_size_k should be a multiple of block_size_k");

  static constexpr int32_t num_block_n = matDst_t::num_block_x;
  static constexpr int32_t num_block_m = matDst_t::num_block_y;
  static constexpr int32_t num_block_k = tile_size_k / block_size_k;

  static constexpr int32_t mma_m =
      register_attr::acc_reg_in_bytes / (block_size_n * sizeof(dtype_dst));

  template <int blk_m, int blk_n, int blk_k>
  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_b, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_m * blk_k> __REF__ a_block) {
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_b, blk_k, blk_n>();
#pragma unroll
    for (uint32_t i = 0; i < blk_m / mma_m; i++) {
      xetla_vector<dtype_dst, mma_m * blk_n> dst_tmp;
      auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, mma_m, blk_n>();
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
        dst_tmp_2d.row(i_acc) = a_block[i_acc + i * mma_m] * b_blk_2d.row(0) +
            src_blk_2d.row(i_acc + i * mma_m);
      }
#pragma unroll
      for (uint32_t k = 1; k < blk_k - 1; k++) {
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          int a_offset = k * blk_m + i_acc + i * mma_m;
          dst_tmp_2d.row(i_acc) += a_block[a_offset] * b_blk_2d.row(k);
        }
      }
      for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
        int a_offset = (blk_k - 1) * blk_m + i_acc + i * mma_m;
        dst_blk_2d.row(i_acc + i * mma_m) =
            a_block[a_offset] * b_blk_2d.row(blk_k - 1) + dst_tmp_2d.row(i_acc);
      }
      SW_BARRIER();
    }

    if constexpr ((blk_m % mma_m) != 0) {
      constexpr uint32_t tail_start_m = blk_m / mma_m * mma_m;
      constexpr uint32_t tail_m = blk_m % mma_m;
      xetla_vector<dtype_dst, tail_m * blk_n> dst_tmp;
      auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, tail_m, blk_n>();
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        dst_tmp_2d.row(i_acc) =
            a_block[i_acc + tail_start_m] * b_blk_2d.row(0) +
            src_blk_2d.row(i_acc + tail_start_m);
      }
#pragma unroll
      for (uint32_t k = 1; k < blk_k - 1; k++) {
        for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
          int a_offset = k * blk_m + i_acc + tail_start_m;
          dst_tmp_2d.row(i_acc) += a_block[a_offset] * b_blk_2d.row(k);
        }
      }
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        int a_offset = (blk_k - 1) * blk_m + i_acc + tail_start_m;
        dst_blk_2d.row(i_acc + tail_start_m) =
            a_block[a_offset] * b_blk_2d.row(blk_k - 1) + dst_tmp_2d.row(i_acc);
      }
    }
  }

  __XETLA_API static void mma(
      matDst_t& dst,
      matSrc_t& src,
      matB_t& b,
      matA_t& a) {
    { // k_blk=0
      auto b_reg = b.reg.xetla_select<b_block_size_y * b_tile_size_x, 1>(0);
#pragma unroll
      for (uint32_t i = 0; i < tile_size_m / block_size_m; i++) {
        auto a_block = a.reg.xetla_select<a_block_elems, 1>(
            i * num_block_k * a_block_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto src_block = src.reg.xetla_select<block_elems, 1>(
              (i * num_block_n + j) * block_elems);
          auto dst_block = dst.reg.xetla_select<block_elems, 1>(
              (i * num_block_n + j) * block_elems);
          mma_core<block_size_m, block_size_n, block_size_k>(
              dst_block, src_block, b_block, a_block);
        }
      }

      // process the tail
      if constexpr ((tile_size_m % block_size_m) != 0) {
        constexpr uint32_t tail_start_m =
            tile_size_m / block_size_m * block_size_m;
        constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
        constexpr uint32_t a_tail_blk_elems = a_block_size_x * a_tail_blk_w;
        constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
        constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;
        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tile_size_x * tail_start_m);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto src_block = src.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          mma_core<tail_size_m, block_size_n, block_size_k>(
              dst_block, src_block, b_block, a_block);
        }
      }
    }
    // different K block
#pragma unroll
    for (uint32_t k_i = 1; k_i < num_block_k; k_i++) {
      auto b_reg = b.reg.xetla_select<b_block_size_y * b_tile_size_x, 1>(
          k_i * b_block_size_y * b_tile_size_x);
#pragma unroll
      for (uint32_t i = 0; i < tile_size_m / block_size_m; i++) {
        auto a_block = a.reg.xetla_select<a_block_elems, 1>(
            (i * num_block_k + k_i) * a_block_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<block_elems, 1>(
              (i * num_block_n + j) * block_elems);
          mma_core<block_size_m, block_size_n, block_size_k>(
              dst_block, dst_block, b_block, a_block);
        }
      }
      // process the tail
      if constexpr ((tile_size_m % block_size_m) != 0) {
        constexpr uint32_t tail_start_m =
            tile_size_m / block_size_m * block_size_m;
        constexpr uint32_t a_tail_blk_w = a_tile_size_y - tail_start_m;
        constexpr uint32_t a_tail_blk_elems = a_block_size_x * a_tail_blk_w;
        constexpr uint32_t tail_size_m = tile_size_m - tail_start_m;
        constexpr uint32_t acc_tail_blk_elems = tail_size_m * block_size_n;
        auto a_block = a.reg.xetla_select<a_tail_blk_elems, 1>(
            a_tile_size_x * tail_start_m + k_i * a_tail_blk_elems);
#pragma unroll
        for (uint32_t j = 0; j < num_block_n; j++) {
          auto b_block =
              b_reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
          auto dst_block = dst.reg.xetla_select<acc_tail_blk_elems, 1>(
              (tail_start_m * tile_size_n) + j * acc_tail_blk_elems);
          mma_core<tail_size_m, block_size_n, block_size_k>(
              dst_block, dst_block, b_block, a_block);
        }
      }
    }
  }
};

} // namespace gpu::xetla::subgroup
