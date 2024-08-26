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

#include <common/core/explicit_conv.hpp>
#include <subgroup/tile/api.hpp>

namespace gpu::xetla::subgroup {

template <
    typename dtype_dst,
    typename dtype_src,
    typename dtype_b,
    typename dtype_a,
    int blk_m,
    int blk_n,
    int blk_k,
    int mma_m,
    reg_layout a_reg_layout,
    reg_layout b_reg_layout,
    mma_engine engine_tag>
struct blk_mma_t {};

template <
    typename dtype_dst,
    typename dtype_src,
    typename dtype_b,
    typename dtype_a,
    int blk_m,
    int blk_n,
    int blk_k,
    int mma_m>
struct blk_mma_t<
    dtype_dst,
    dtype_src,
    dtype_b,
    dtype_a,
    blk_m,
    blk_n,
    blk_k,
    mma_m,
    reg_layout::transpose_tiled,
    reg_layout::tiled,
    mma_engine::fpu> {
  static constexpr uint32_t blk_m_iters = blk_m / mma_m;
  static constexpr uint32_t tail_m = blk_m % mma_m;
  static constexpr uint32_t tail_start_m = blk_m_iters * mma_m;
  static constexpr uint32_t a_block_elems = blk_m * blk_k;

  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_b, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_k * blk_m> __REF__ a_block) {
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_src, blk_k, blk_n>();
    xetla_vector<dtype_src, a_block_elems> new_a_block =
        xetla_cvt<dtype_src, dtype_a, a_block_elems>(
            a_block.xetla_select<a_block_elems, 1>(0));

    if constexpr (blk_m_iters > 0) {
#pragma unroll
      for (uint32_t i = 0; i < blk_m_iters; i++) {
        xetla_vector<dtype_dst, mma_m * blk_n> dst_tmp;
        auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, mma_m, blk_n>();
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          dst_tmp_2d.row(i_acc) =
              new_a_block[i_acc + i * mma_m] * b_blk_2d.row(0) +
              src_blk_2d.row(i_acc + i * mma_m);
        }
#pragma unroll
        for (uint32_t k = 1; k < blk_k - 1; k++) {
          auto b_blk_k = b_blk_2d.row(k);
#pragma unroll
          for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
            int a_offset = k * blk_m + i_acc + i * mma_m;
            dst_tmp_2d.row(i_acc) += new_a_block[a_offset] * b_blk_k;
          }
        }
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          int a_offset = (blk_k - 1) * blk_m + i_acc + i * mma_m;
          dst_blk_2d.row(i_acc + i * mma_m) =
              new_a_block[a_offset] * b_blk_2d.row(blk_k - 1) +
              dst_tmp_2d.row(i_acc);
        }
      }
    }

    if constexpr (tail_m != 0) {
      xetla_vector<dtype_dst, tail_m * blk_n> dst_tmp;
      auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, tail_m, blk_n>();
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        dst_tmp_2d.row(i_acc) =
            new_a_block[i_acc + tail_start_m] * b_blk_2d.row(0) +
            src_blk_2d.row(i_acc + tail_start_m);
      }
#pragma unroll
      for (uint32_t k = 1; k < blk_k - 1; k++) {
        auto b_blk_k = b_blk_2d.row(k);
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
          int a_offset = k * blk_m + i_acc + tail_start_m;
          dst_tmp_2d.row(i_acc) += new_a_block[a_offset] * b_blk_k;
        }
      }
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        int a_offset = (blk_k - 1) * blk_m + i_acc + tail_start_m;
        dst_blk_2d.row(i_acc + tail_start_m) =
            new_a_block[a_offset] * b_blk_2d.row(blk_k - 1) +
            dst_tmp_2d.row(i_acc);
      }
    }
  }
};

template <
    typename dtype_dst,
    typename dtype_src,
    typename dtype_b,
    typename dtype_a,
    int blk_m,
    int blk_n,
    int blk_k,
    int mma_m>
struct blk_mma_t<
    dtype_dst,
    dtype_src,
    dtype_b,
    dtype_a,
    blk_m,
    blk_n,
    blk_k,
    mma_m,
    reg_layout::tiled,
    reg_layout::tiled,
    mma_engine::fpu> {
  static constexpr uint32_t blk_m_iters = blk_m / mma_m;
  static constexpr uint32_t tail_m = blk_m % mma_m;
  static constexpr uint32_t tail_start_m = blk_m_iters * mma_m;
  static constexpr uint32_t a_block_elems = blk_m * blk_k;

  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_b, blk_k * blk_n> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_m * blk_k> __REF__ a_block) {
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_src, blk_k, blk_n>();
    xetla_vector<dtype_src, a_block_elems> new_a_block =
        xetla_cvt<dtype_src, dtype_a, a_block_elems>(
            a_block.xetla_select<a_block_elems, 1>(0));

    if constexpr (blk_m_iters > 0) {
#pragma unroll
      for (uint32_t i = 0; i < blk_m_iters; i++) {
        auto b_blk_k0 = b_blk_2d.row(0);
        int32_t a_start_off = i * mma_m * blk_k;
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          dst_blk_2d.row(i_acc + i * mma_m) =
              src_blk_2d.row(i_acc + i * mma_m) +
              new_a_block[a_start_off + i_acc * blk_k] * b_blk_k0;
        }

#pragma unroll
        for (uint32_t k = 1; k < blk_k; k++) {
          auto b_blk_k = b_blk_2d.row(k);
#pragma unroll
          for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
            dst_blk_2d.row(i_acc + i * mma_m) +=
                new_a_block[a_start_off + i_acc * blk_k + k] * b_blk_k;
          }
        }
      }
    }

    if constexpr (tail_m != 0) {
      auto b_blk_k0 = b_blk_2d.row(0);
      int32_t a_start_off = tail_start_m * blk_k;
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        dst_blk_2d.row(i_acc + tail_start_m) =
            src_blk_2d.row(i_acc + tail_start_m) +
            new_a_block[a_start_off + i_acc * blk_k] * b_blk_k0;
      }
#pragma unroll
      for (uint32_t k = 1; k < blk_k; k++) {
        auto b_blk_k = b_blk_2d.row(k);
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
          dst_blk_2d.row(i_acc + tail_start_m) +=
              new_a_block[a_start_off + i_acc * blk_k + k] * b_blk_k;
        }
      }
    }
  }
};

template <
    typename dtype_dst,
    typename dtype_src,
    typename dtype_b,
    typename dtype_a,
    int blk_m,
    int blk_n,
    int blk_k,
    int mma_m>
struct blk_mma_t<
    dtype_dst,
    dtype_src,
    dtype_b,
    dtype_a,
    blk_m,
    blk_n,
    blk_k,
    mma_m,
    reg_layout::tiled,
    reg_layout::transpose_tiled,
    mma_engine::fpu> {
  static constexpr uint32_t blk_m_iters = blk_m / mma_m;
  static constexpr uint32_t tail_m = blk_m % mma_m;
  static constexpr uint32_t a_block_elems = blk_m * blk_k;
  static constexpr uint32_t tail_start_m = blk_m_iters * mma_m;

  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_b, blk_n * blk_k> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_m * blk_k> __REF__ a_block) {
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_src, blk_n, blk_k>();
    xetla_vector<dtype_src, a_block_elems> new_a_block =
        xetla_cvt<dtype_src, dtype_a, a_block_elems>(
            a_block.xetla_select<a_block_elems, 1>(0));
    auto a_blk_2d = new_a_block.xetla_format<dtype_src, blk_m, blk_k>();

    if constexpr (blk_m_iters > 0) {
#pragma unroll
      for (uint32_t i = 0; i < blk_m_iters; i++) {
        auto i_start_m = i * mma_m;
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          auto moffset = i_acc + i_start_m;
          auto dst_row = dst_blk_2d.row(moffset);
          auto src_row = src_blk_2d.row(moffset);
          auto a_blk_m = a_blk_2d.row(moffset);
          xetla_vector<dtype_src, blk_k> tmp_k = a_blk_m * b_blk_2d.row(0);
          dst_row[0] = src_row[0] +
              xetla_reduce<dtype_src, dtype_src, blk_k, reduce_op::sum>(tmp_k);
#pragma unroll
          for (uint32_t j = 1; j < blk_n; j++) {
            tmp_k = a_blk_m * b_blk_2d.row(j);
            dst_row[j] = src_row[j] +
                xetla_reduce<dtype_src, dtype_src, blk_k, reduce_op::sum>(
                             tmp_k);
          }
        }
      }
    }

    if constexpr (tail_m != 0) {
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        auto moffset = i_acc + tail_start_m;
        auto dst_row = dst_blk_2d.row(moffset);
        auto src_row = src_blk_2d.row(moffset);
        auto a_blk_m = a_blk_2d.row(moffset);
        xetla_vector<dtype_src, blk_k> tmp_k = a_blk_m * b_blk_2d.row(0);
        dst_row[0] = src_row[0] +
            xetla_reduce<dtype_src, dtype_src, blk_k, reduce_op::sum>(tmp_k);
#pragma unroll
        for (uint32_t j = 1; j < blk_n; j++) {
          tmp_k = a_blk_m * b_blk_2d.row(j);
          dst_row[j] = src_row[j] +
              xetla_reduce<dtype_src, dtype_src, blk_k, reduce_op::sum>(tmp_k);
        }
      }
    }
  }
};

template <
    typename dtype_dst,
    typename dtype_src,
    typename dtype_b,
    typename dtype_a,
    int blk_m,
    int blk_n,
    int blk_k,
    int mma_m>
struct blk_mma_t<
    dtype_dst,
    dtype_src,
    dtype_b,
    dtype_a,
    blk_m,
    blk_n,
    blk_k,
    mma_m,
    reg_layout::transpose_tiled,
    reg_layout::transpose_tiled,
    mma_engine::fpu> {
  static constexpr uint32_t blk_m_iters = blk_m / mma_m;
  static constexpr uint32_t tail_m = blk_m % mma_m;
  static constexpr uint32_t a_block_elems = blk_m * blk_k;
  static constexpr uint32_t tail_start_m = blk_m_iters * mma_m;

  __XETLA_API static void mma_core(
      xetla_vector_ref<dtype_dst, blk_m * blk_n> __REF__ dst,
      xetla_vector_ref<dtype_src, blk_m * blk_n> __REF__ src,
      xetla_vector_ref<dtype_b, blk_n * blk_k> __REF__ b_block,
      xetla_vector_ref<dtype_a, blk_k * blk_m> __REF__ a_block) {
    auto dst_blk_2d = dst.xetla_format<dtype_dst, blk_m, blk_n>();
    auto src_blk_2d = src.xetla_format<dtype_src, blk_m, blk_n>();
    auto b_blk_2d = b_block.xetla_format<dtype_src, blk_n, blk_k>();
    xetla_vector<dtype_src, a_block_elems> new_a_block =
        xetla_cvt<dtype_src, dtype_a, a_block_elems>(
            a_block.xetla_select<a_block_elems, 1>(0));

    if constexpr (blk_m_iters > 0) {
#pragma unroll
      for (uint32_t i = 0; i < blk_m_iters; i++) {
        xetla_vector<dtype_dst, mma_m * blk_n> dst_tmp;
        auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, mma_m, blk_n>();
        auto b_blk_k0 = b_blk_2d.column(0);
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          dst_tmp_2d.row(i_acc) = new_a_block[i_acc + i * mma_m] * b_blk_k0 +
              src_blk_2d.row(i_acc + i * mma_m);
        }
#pragma unroll
        for (uint32_t k = 1; k < blk_k - 1; k++) {
          auto b_blk_k = b_blk_2d.column(k);
#pragma unroll
          for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
            int a_offset = k * blk_m + i_acc + i * mma_m;
            dst_tmp_2d.row(i_acc) += new_a_block[a_offset] * b_blk_k;
          }
        }
        auto b_blk_k_last = b_blk_2d.column(blk_k - 1);
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < mma_m; i_acc++) {
          int a_offset = (blk_k - 1) * blk_m + i_acc + i * mma_m;
          dst_blk_2d.row(i_acc + i * mma_m) =
              new_a_block[a_offset] * b_blk_k_last + dst_tmp_2d.row(i_acc);
        }
      }
    }

    if constexpr (tail_m != 0) {
      xetla_vector<dtype_dst, tail_m * blk_n> dst_tmp;
      auto dst_tmp_2d = dst_tmp.xetla_format<dtype_dst, tail_m, blk_n>();
      auto b_blk_k0 = b_blk_2d.column(0);
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        dst_tmp_2d.row(i_acc) = new_a_block[i_acc + tail_start_m] * b_blk_k0 +
            src_blk_2d.row(i_acc + tail_start_m);
      }
#pragma unroll
      for (uint32_t k = 1; k < blk_k - 1; k++) {
        auto b_blk_k = b_blk_2d.column(k);
#pragma unroll
        for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
          int a_offset = k * blk_m + i_acc + tail_start_m;
          dst_tmp_2d.row(i_acc) += new_a_block[a_offset] * b_blk_k;
        }
      }
      auto b_blk_k_last = b_blk_2d.column(blk_k - 1);
#pragma unroll
      for (uint32_t i_acc = 0; i_acc < tail_m; i_acc++) {
        int a_offset = (blk_k - 1) * blk_m + i_acc + tail_start_m;
        dst_blk_2d.row(i_acc + tail_start_m) =
            new_a_block[a_offset] * b_blk_k_last + dst_tmp_2d.row(i_acc);
      }
    }
  }
};

} // namespace gpu::xetla::subgroup
