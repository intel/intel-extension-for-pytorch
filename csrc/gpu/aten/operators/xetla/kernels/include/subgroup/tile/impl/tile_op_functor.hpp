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
#include <subgroup/tile/common.hpp>
#include <subgroup/tile/impl/load_xe.hpp>
#include <subgroup/tile/impl/payload_xe.hpp>
#include <subgroup/tile/impl/prefetch_xe.hpp>
#include <subgroup/tile/impl/reduction.hpp>
#include <subgroup/tile/impl/store_xe.hpp>

namespace gpu::xetla::subgroup {

/// @brief Is none op functor, for placeholder purpose.
/// Used in epilogue::tile_op or chained_tile_op.
struct none_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      [[maybe_unused]] matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {}
  // none_op_t functor for dequant_op
  template <typename mat_out_t, typename mat_in_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      [[maybe_unused]] mat_out_t& mat_out,
      [[maybe_unused]] mat_in_t& mat_in,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args) {
    mat_out = mat_in;
  }
};

template <
    typename matB_acc_t,
    typename matB_t,
    typename scale_t,
    typename zero_pt_t,
    uint32_t dequant_s,
    quant_mode quant_mode>
struct dequant_int4_weight_t {
  struct arguments_t {
    uint32_t wg_start_m;
    uint32_t wg_start_n;
    uint32_t wg_start_k;
    inline arguments_t() = default;
    inline arguments_t(uint32_t wg_start_n_, uint32_t wg_start_k_)
        : wg_start_n(wg_start_n_), wg_start_k(wg_start_k_) {}
  };
  __XETLA_API KERNEL_FUNC void operator()(
      matB_acc_t& matB_acc,
      matB_t& matB,
      scale_t& scale,
      zero_pt_t& zero_pt,
      //   [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    // no tail, because this is matB
    constexpr uint32_t tile_size_x_b = matB_acc_t::tile_size_x;
    constexpr uint32_t tile_size_y_b = matB_acc_t::tile_size_y;
    constexpr uint32_t block_size_x_b = matB_acc_t::block_size_x;
    constexpr uint32_t block_size_y_b = matB_acc_t::block_size_y;
    static constexpr uint32_t pack_ratio = sizeof(typename matB_t::dtype) * 2;

    constexpr uint32_t num_block_x = tile_size_x_b / block_size_x_b;
    constexpr uint32_t num_block_y = tile_size_y_b / block_size_y_b;
#pragma unroll
    for (uint32_t i = 0; i < num_block_y; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        int block_id = (i * num_block_x + j);
        // Must be little-endian
        auto matB_blk = matB.reg.xetla_format<uint8_t>()
                            .xetla_select<matB_acc_t::block_elems / 2, 1>(
                                block_id * matB_acc_t::block_elems / 2);

        auto dst_blk = matB_acc.reg.xetla_select<matB_acc_t::block_elems, 1>(
            block_id * matB_acc_t::block_elems);

        // int8 includes 2 4bits data.
        xetla_vector<int8_t, matB_acc_t::block_elems> cvt_blk_i8;

        // lowest 4 bit
        {
          cvt_blk_i8.xetla_select<matB_acc_t::block_elems / 2, 2>(0) =
              matB_blk & 0xf;
        }
        // highest 4 bit
        {
          cvt_blk_i8.xetla_select<matB_acc_t::block_elems / 2, 2>(1) =
              matB_blk >> 4;
        }

        // (b_i8 -  zero_pt_i8) x scale = fp16
        constexpr uint32_t step = std::min(block_size_y_b, dequant_s);
#pragma unroll
        for (uint32_t jj = 0; jj < block_size_x_b; jj++) {
#pragma unroll
          for (uint32_t ii = 0; ii < block_size_y_b; ii += step) {
            uint32_t offset_y_in_tile = i * block_size_y_b + ii;
            uint32_t offset_x_in_tile = j * block_size_x_b + jj;

            uint32_t scale_idx =
                (offset_y_in_tile) / dequant_s * scale_t::block_size_x +
                offset_x_in_tile;

            if constexpr (quant_mode == quant_mode::I4_ASYM) {
              uint32_t zero_pt_idx =
                  offset_y_in_tile / dequant_s * zero_pt_t::block_size_x +
                  offset_x_in_tile / pack_ratio;
              native_type_t<typename matB_t::dtype> zero_pt_pack =
                  zero_pt.reg[zero_pt_idx];

              int8_t zero_pt_i8 =
                  (zero_pt_pack >>
                   (4 * ((args.wg_start_n + offset_x_in_tile) % pack_ratio))) &
                  0xf;
              // sycl::ext::oneapi::experimental::printf(
              //     "zero_pt.reg[%d}  %x    zero_pt_i8 %x  offset_x_in_tile:%d
              //     \n", zero_pt_idx, zero_pt_pack, (int32_t)zero_pt_i8 ,
              //     offset_x_in_tile);

              cvt_blk_i8.xetla_select<step, 1>(jj * block_size_y_b + ii) =
                  cvt_blk_i8.xetla_select<step, 1>(jj * block_size_y_b + ii) -
                  zero_pt_i8;
            } else if constexpr (
                quant_mode == quant_mode::I4_SYM ||
                quant_mode == quant_mode::I4_ASYM_FP_ZERO) {
              cvt_blk_i8.xetla_select<step, 1>(jj * block_size_y_b + ii) =
                  cvt_blk_i8.xetla_select<step, 1>(jj * block_size_y_b + ii) -
                  int8_t(8);
            }
            dst_blk.xetla_select<step, 1>(jj * block_size_y_b + ii) =
                cvt_blk_i8.xetla_select<step, 1>(jj * block_size_y_b + ii) *
                scale.reg[scale_idx];
            if constexpr (quant_mode == quant_mode::I4_ASYM_FP_ZERO) {
              uint32_t zero_pt_idx =
                  offset_y_in_tile / dequant_s * zero_pt_t::block_size_x +
                  offset_x_in_tile;
              // xetla_vector<fp16, 1> zero_pt_pack = zero_pt.reg[zero_pt_idx];
              // dst_blk.xetla_select<step, 1>(jj * block_size_y_b + ii) =
              //     dst_blk.xetla_select<step, 1>(jj * block_size_y_b + ii) +
              //     zero_pt_pack[0];
              dst_blk.xetla_select<step, 1>(jj * block_size_y_b + ii) =
                  dst_blk.xetla_select<step, 1>(jj * block_size_y_b + ii) +
                  zero_pt.reg[zero_pt_idx];
            }
            // sycl::ext::oneapi::experimental::printf(
            //     "scale[%d] %f \n",
            //     scale_idx,
            //     float(sycl::half(scale.reg.xetla_select<1, 1>(scale_idx))));
          }
        }
      }
    }
  }
};

template <int N, fp8_format fmt>
__XETLA_API typename std::
    enable_if_t<fmt == fp8_format::E5M2, xetla_vector<sycl::half, N>>
    fp8_to_fp16(xetla_vector<uint8_t, N> src) {
  // Do not consider NAN
  xetla_vector<uint16_t, N> bits_fp16 = src << 8;
  return bits_fp16.template bit_cast_view<sycl::half>();
}

template <int N, fp8_format fmt>
__XETLA_API typename std::
    enable_if_t<fmt == fp8_format::E4M3, xetla_vector<sycl::half, N>>
    fp8_to_fp16(xetla_vector<uint8_t, N> src) {
  // Do not consider NAN or denormal
  xetla_vector<uint8_t, N> s8 = src >> 7;
  xetla_vector<uint8_t, N> e8 = ((src & 0x78) >> 3) + 8;
  xetla_vector<uint8_t, N> m8 = src & 0x7;
  auto is_zero = ((e8 == 0) & (m8 == 0));
  e8.merge(xetla_vector<uint8_t, N>(0), is_zero);
  xetla_vector<uint16_t, N> bits_fp16 = (s8 << 15) | (e8 << 10) | (m8 << 7);
  return bits_fp16.template bit_cast_view<sycl::half>();
}

template <typename matB_acc_t, typename matB_t, fp8_format fp8_format>
struct dequant_fp8_weight_t {
  using dtype_dst = typename matB_acc_t::dtype;
  using dtype_src = typename matB_t::dtype;

  __XETLA_API KERNEL_FUNC void operator()(
      matB_acc_t& dst,
      matB_t& src,
      float scale) {
    constexpr uint32_t tile_size_y = matB_acc_t::tile_size_y;
    constexpr uint32_t tile_size_x = matB_acc_t::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = matB_acc_t::block_size_y;
    constexpr uint32_t block_size_x = matB_acc_t::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    assert(typeid(dtype_src) == typeid(uint8_t));
    constexpr uint32_t vnni_row_src = sizeof(uint32_t) / sizeof(uint8_t);
    constexpr uint32_t vnni_row_dst = sizeof(uint32_t) / sizeof(dtype_dst);
    constexpr int32_t vnni_row =
        vnni_row_src > vnni_row_dst ? vnni_row_src : vnni_row_dst;
    static_assert(block_size_y % vnni_row == 0);
    static_assert(tile_size_y % vnni_row == 0);
    constexpr int32_t move_elems = vnni_row * block_size_x;
    xetla_vector<dtype_dst, tile_elems> reg_src =
        xetla_cvt<dtype_dst, sycl::half, tile_elems>(
            fp8_to_fp16<tile_elems, fp8_format>(src.reg));
    if constexpr (sizeof(uint8_t) == sizeof(dtype_dst)) {
      dst.reg = reg_src;
      return;
    }
    xetla_vector<dtype_dst, tile_elems> reg_dst;
    constexpr uint32_t scale_factor =
        detail::gcd<vnni_row_src, vnni_row_dst>::value;
    using move_dtype = get_uint_type_t<sizeof(dtype_dst) * scale_factor>;
    constexpr uint32_t select_stride = vnni_row / scale_factor;
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto reg_src_blk = reg_src.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto reg_dst_blk = reg_dst.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        for (uint32_t row_i = 0; row_i < block_size_y; row_i += vnni_row) {
          auto reg_src_move =
              reg_src_blk.xetla_select<move_elems, 1>(row_i * block_size_x)
                  .xetla_format<native_type_t<move_dtype>>();
          auto reg_dst_move =
              reg_dst_blk.xetla_select<move_elems, 1>(row_i * block_size_x)
                  .xetla_format<native_type_t<move_dtype>>();
#pragma unroll
          for (uint32_t move_i = 0; move_i < select_stride; move_i++) {
            if constexpr (sizeof(dtype_dst) > sizeof(uint8_t)) {
              reg_dst_move.xetla_select<block_size_x, 1>(
                  move_i * block_size_x) =
                  reg_src_move.xetla_select<block_size_x, select_stride>(
                      move_i);
            } else {
              reg_dst_move.xetla_select<block_size_x, select_stride>(move_i) =
                  reg_src_move.xetla_select<block_size_x, 1>(
                      move_i * block_size_x);
            }
          }
        }
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr int i = tile_size_y / block_size_y;
      constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
      constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
      constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto reg_src_blk = reg_src.xetla_select<remain_block_elems, 1>(
            remain_elems_start + j * remain_block_elems);
        auto reg_dst_blk = reg_dst.xetla_select<remain_block_elems, 1>(
            remain_elems_start + j * remain_block_elems);
        // for mma, here we can guarantee that the remaining is a multiple of
        // vnni_row
        for (uint32_t row_i = 0; row_i < remain_size_y; row_i += vnni_row) {
          auto reg_src_move =
              reg_src_blk.xetla_select<move_elems, 1>(row_i * block_size_x)
                  .xetla_format<native_type_t<move_dtype>>();
          auto reg_dst_move =
              reg_dst_blk.xetla_select<move_elems, 1>(row_i * block_size_x)
                  .xetla_format<native_type_t<move_dtype>>();
#pragma unroll
          for (uint32_t move_i = 0; move_i < select_stride; move_i++) {
            if constexpr (sizeof(dtype_dst) > sizeof(uint8_t)) {
              reg_dst_move.xetla_select<block_size_x, 1>(
                  move_i * block_size_x) =
                  reg_src_move.xetla_select<block_size_x, select_stride>(
                      move_i);
            } else {
              reg_dst_move.xetla_select<block_size_x, select_stride>(move_i) =
                  reg_src_move.xetla_select<block_size_x, 1>(
                      move_i * block_size_x);
            }
          }
        }
      }
    }
    dst.reg = reg_dst;
  }
};

/// @brief Is the element-wise relu op functor.
/// Get the relu input from matAcc, update the relu output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct relu_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    xetla_mask<matAcc_t::tile_elems> mask = matAcc.reg < 0;
    matAcc.reg.xetla_merge(0, mask);
  }
};

/// @brief Is the element-wise tanh op functor.
/// Get the tanh input from matAcc, update the the tanh output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct tanh_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    constexpr int elems = matAcc_t::tile_desc::block_elems;
    constexpr int rounds = matAcc_t::tile_desc::tile_elems / elems;
    using dtype = typename matAcc_t::dtype;
#pragma unroll
    for (uint32_t i = 0; i < rounds; ++i) {
      auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
      sub_vec = xetla_tanh<dtype, elems>(sub_vec);
    }
    constexpr int remaining_elems = matAcc_t::tile_desc::tile_elems % elems;
    if constexpr (remaining_elems != 0) {
      auto sub_vec = matAcc.reg.xetla_select<remaining_elems, 1>(
          elems * (matAcc_t::tile_elems / elems));
      sub_vec = xetla_tanh<dtype, remaining_elems>(sub_vec);
    }
  }
};

/// @brief Is the element-wise sigmoid op functor.
/// Get the sigmoid input from matAcc, update the the sigmoid output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct sigmoid_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    constexpr int elems = matAcc_t::tile_desc::block_elems;
    constexpr int rounds = matAcc_t::tile_desc::tile_elems / elems;
#pragma unroll
    for (uint32_t i = 0; i < rounds; ++i) {
      auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
      sub_vec = xetla_sigmoid<typename matAcc_t::dtype, elems>(sub_vec);
    }
    constexpr int remaining_elems = matAcc_t::tile_desc::tile_elems % elems;
    if constexpr (remaining_elems != 0) {
      auto sub_vec = matAcc.reg.xetla_select<remaining_elems, 1>(
          elems * (matAcc_t::tile_elems / elems));
      sub_vec =
          xetla_sigmoid<typename matAcc_t::dtype, remaining_elems>(sub_vec);
    }
  }
};

/// @brief Is the element-wise silu op functor.
/// Get the silu input from matAcc, update the the silu output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct silu_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      [[maybe_unused]] matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    constexpr int elems = matAcc_t::tile_desc::block_elems;
    constexpr int rounds = matAcc_t::tile_desc::tile_elems / elems;
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
      auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
      sub_vec = xetla_silu<typename matAcc_t::dtype, elems>(sub_vec);
    }
    constexpr int remaining_elems = matAcc_t::tile_desc::tile_elems % elems;
    if constexpr (remaining_elems != 0) {
      auto sub_vec = matAcc.reg.xetla_select<remaining_elems, 1>(
          elems * (matAcc_t::tile_elems / elems));
      sub_vec = xetla_silu<typename matAcc_t::dtype, remaining_elems>(sub_vec);
    }
  }
};

/// @brief Is the element-wise gelu inference forward op functor.
/// Get the gelu input from matAcc, update the the gelu output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct gelu_fwd_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype = typename matAcc_t::dtype;
    constexpr dtype C0 = 0.044715f;
    constexpr dtype sqrt_two_over_pi = 0.79788458347320556640625f;
    // total flag register
    constexpr int elems = 8 * 16;
    constexpr int rounds = matAcc_t::tile_elems / elems;
    if constexpr (rounds != 0) {
#pragma unroll
      for (int i = 0; i < rounds; ++i) {
        auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
        xetla_vector<dtype, elems> sub_vec_x =
            (sqrt_two_over_pi * sub_vec * (1.f + C0 * sub_vec * sub_vec));
        xetla_vector<dtype, elems> tanh_value =
            xetla_tanh<dtype, elems>(sub_vec_x);
        sub_vec = 0.5f * sub_vec * (1.f + tanh_value);
      }
    }

    constexpr int remaining_elems = matAcc_t::tile_elems % elems;
    if constexpr (remaining_elems != 0) {
      auto sub_vec = matAcc.reg.xetla_select<remaining_elems, 1>(
          elems * (matAcc_t::tile_elems / elems));
      xetla_vector<dtype, remaining_elems> sub_vec_x =
          (sqrt_two_over_pi * sub_vec * (1.f + C0 * sub_vec * sub_vec));
      xetla_vector<dtype, remaining_elems> tanh_value =
          xetla_tanh<dtype, remaining_elems>(sub_vec_x);
      sub_vec = 0.5f * sub_vec * (1.f + tanh_value);
    }
  }
};

/// @brief Is the element-wise gelu training forward op functor.
/// Get the gelu input from matAcc, update the the gelu output in place,
/// and dump the intermediate buffer_w to memory for backward purpose.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_out Is the data type of the intermediate buffer_w.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_out, gpu_arch arch_tag, class enable = void>
struct gelu_fwd_w_op_t {};
/// @brief Is the element-wise gelu training forward op functor, specialized
/// for Xe architecture.
template <typename dtype_out_, gpu_arch arch_tag>
struct gelu_fwd_w_op_t<
    dtype_out_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_out = dtype_out_;
  using mem_desc_w_t =
      mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_w_t::shape_t;
  using coord_t = typename mem_desc_w_t::coord_t;
  using base_t = typename mem_desc_w_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : shape(shape_), base(base_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;

    mem_desc_w_t mem_desc_w(args.base, args.shape, coord);
    using bwd_w_tile_desc_t = tile_desc_t<
        block_size_x,
        block_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using bwd_w_tile_t = tile_t<dtype_out, bwd_w_tile_desc_t>;
    using bwd_w_payload_t = mem_payload_t<
        mem_desc_w_t,
        bwd_w_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    bwd_w_tile_t bwd_w;
    bwd_w_payload_t bwd_w_payload(mem_desc_w);
    // start compute
    constexpr dtype_acc c0 = 0.044715f;
    constexpr dtype_acc d0 = 0.134145f;
    constexpr dtype_acc sqrt_two_over_pi = 0.79788458347320556640625f;
    constexpr uint32_t block_elems = matAcc_t::block_elems;
    constexpr uint32_t num_block_x = matAcc_t::num_block_x;
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        auto x = matAcc.reg.xetla_select<block_elems, 1>(
            block_elems * (i * num_block_x + j));
        xetla_vector<dtype_acc, block_elems> z =
            xetla_tanh<dtype_acc, block_elems>(
                sqrt_two_over_pi * (x + c0 * x * x * x));
        xetla_vector<dtype_acc, block_elems> w =
            (0.5f * (1.f + z) +
             0.5f * x * (1.f - z * z) *
                 (sqrt_two_over_pi * (1.f + d0 * x * x)));
        x = 0.5f * x * (1.f + z);
        bwd_w.reg = xetla_cvt<dtype_out, dtype_acc, block_elems>(w);
        tile_store<cache_hint::uncached>(bwd_w, bwd_w_payload);
        bwd_w_payload.template update_tdesc<tdesc_update_dir::x_dir>(
            block_size_x);
      }
      bwd_w_payload.template update_tdesc<tdesc_update_dir::x_dir>(
          -1 * tile_size_x);
      bwd_w_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          block_size_y);
    }
    if constexpr (tile_size_y % block_size_y != 0) {
      constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
      constexpr uint32_t remain_y_start =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t remain_elems_start = remain_y_start * tile_size_x;
      constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;

      using remain_bwd_w_tile_desc_t = tile_desc_t<
          block_size_x,
          remain_size_y,
          block_size_x,
          remain_size_y,
          reg_layout::tiled>;
      using remain_bwd_w_tile_t = tile_t<dtype_out, remain_bwd_w_tile_desc_t>;
      using remain_bwd_w_payload_t = mem_payload_t<
          mem_desc_w_t,
          remain_bwd_w_tile_desc_t,
          msg_type::block_2d,
          arch_tag>;

      mem_desc_w.update_coord_y(remain_y_start);
      remain_bwd_w_payload_t remain_bwd_w_payload(mem_desc_w);
      remain_bwd_w_tile_t remain_bwd_w;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        auto x = matAcc.reg.xetla_select<remain_block_elems, 1>(
            remain_elems_start + remain_block_elems * j);
        xetla_vector<dtype_acc, remain_block_elems> z =
            xetla_tanh<dtype_acc, remain_block_elems>(
                sqrt_two_over_pi * (x + c0 * x * x * x));
        xetla_vector<dtype_acc, remain_block_elems> w =
            (0.5f * (1.f + z) +
             0.5f * x * (1.f - z * z) *
                 (sqrt_two_over_pi * (1.f + d0 * x * x)));
        x = 0.5f * x * (1.f + z);
        remain_bwd_w.reg =
            xetla_cvt<dtype_out, dtype_acc, remain_block_elems>(w);
        tile_store<cache_hint::uncached>(remain_bwd_w, remain_bwd_w_payload);
        remain_bwd_w_payload.template update_tdesc<tdesc_update_dir::x_dir>(
            block_size_x);
      }
    }
  }
};

/// @brief Is the element-wise gelu backward op functor.
/// Load the gelu forward input buffer from memory and get the gradient data
/// from matAcc, update the output in place. Used in epilogue::tile_op or
/// chained_tile_op.
/// @tparam dtype_in Is the data type of the gelu forward input buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_in, gpu_arch arch_tag, class enable = void>
struct gelu_bwd_op_t {};
/// @brief Is the element-wise gelu backward op functor, specialized for Xe
/// architecture.
template <typename dtype_in_, gpu_arch arch_tag>
struct gelu_bwd_op_t<
    dtype_in_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_in = dtype_in_;
  using mem_desc_x_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_x_t::shape_t;
  using coord_t = typename mem_desc_x_t::coord_t;
  using base_t = typename mem_desc_x_t::base_t;
  struct arguments_t {
    shape_t shape;
    base_t base;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : shape(shape_), base(base_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;

    using bwd_x_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using bwd_x_tile_t = tile_t<dtype_in, bwd_x_tile_desc_t>;
    using bwd_x_payload_t = mem_payload_t<
        mem_desc_x_t,
        bwd_x_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    bwd_x_tile_t bwd_x;
    // init tdesc
    mem_desc_x_t mem_desc_x(args.base, args.shape, coord);
    bwd_x_payload_t bwd_x_payload(mem_desc_x);
    tile_load<cache_hint::cached, cache_hint::cached>(bwd_x, bwd_x_payload);
    // start compute
    constexpr dtype_acc c0 = 0.044715f;
    constexpr dtype_acc d0 = 0.134145f;
    constexpr dtype_acc sqrt_two_over_pi = 0.79788458347320556640625f;
    constexpr uint32_t block_elems = matAcc_t::block_elems;
    constexpr uint32_t num_block_x = matAcc_t::num_block_x;
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        auto x_in = bwd_x.reg.xetla_select<block_elems, 1>(
            block_elems * (i * num_block_x + j));
        auto x = xetla_cvt<dtype_acc, dtype_in, block_elems>(x_in);
        auto dy = matAcc.reg.xetla_select<block_elems, 1>(
            block_elems * (i * num_block_x + j));
        xetla_vector<dtype_acc, block_elems> z =
            xetla_tanh<dtype_acc, block_elems>(
                sqrt_two_over_pi * (x + c0 * x * x * x));
        xetla_vector<dtype_acc, block_elems> w =
            (0.5f * (1.f + z) +
             0.5f * x * (1.f - z * z) *
                 (sqrt_two_over_pi * (1.f + d0 * x * x)));
        dy = w * dy;
      }
    }
    if constexpr (tile_size_y % block_size_y != 0) {
      constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
      constexpr uint32_t remain_y_start =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t remain_elems_start = remain_y_start * tile_size_x;
      constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        auto x_in = bwd_x.reg.xetla_select<remain_block_elems, 1>(
            remain_elems_start + remain_block_elems * j);
        auto x = xetla_cvt<dtype_acc, dtype_in, remain_block_elems>(x_in);
        auto dy = matAcc.reg.xetla_select<remain_block_elems, 1>(
            remain_elems_start + remain_block_elems * j);
        xetla_vector<dtype_acc, remain_block_elems> z =
            xetla_tanh<dtype_acc, remain_block_elems>(
                sqrt_two_over_pi * (x + c0 * x * x * x));
        xetla_vector<dtype_acc, remain_block_elems> w =
            (0.5f * (1.f + z) +
             0.5f * x * (1.f - z * z) *
                 (sqrt_two_over_pi * (1.f + d0 * x * x)));
        dy = w * dy;
      }
    }
  }
};

/// @brief Is the bias_add op functor.
/// Load the 1d bias data from memory and get the input from matAcc, update
/// the output in place. Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_bias Is the data type of bias buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename mem_desc_bias_t_, gpu_arch arch_tag, class enable = void>
struct bias_add_op_t {};
/// @brief Is the bias_add op functor, specialized for Xe architecture.
template <typename mem_desc_bias_t_, gpu_arch arch_tag>
struct bias_add_op_t<
    mem_desc_bias_t_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using mem_desc_bias_t = mem_desc_bias_t_;
  using dtype_bias = typename mem_desc_bias_t::dtype;
  using shape_t = typename mem_desc_bias_t::shape_t;
  using coord_t = typename mem_desc_bias_t::coord_t;
  using base_t = typename mem_desc_bias_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : shape(shape_), base(base_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using bias_tile_desc_t =
        tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
    using bias_t = tile_t<dtype_bias, bias_tile_desc_t>;
    using bias_payload_t = mem_payload_t<
        mem_desc_bias_t,
        bias_tile_desc_t,
        msg_type::block_2d,
        arch_tag>;
    coord_t bias_coord(coord.x, 0);
    mem_desc_bias_t mem_desc_bias(args.base, args.shape, bias_coord);
    bias_t bias;
    bias_payload_t bias_payload(mem_desc_bias);
    tile_load<cache_hint::cached, cache_hint::cached>(bias, bias_payload);

#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto dst_reg =
            matAcc.reg
                .xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems)
                .xetla_format<dtype_acc, block_size_y, block_size_x>();
#pragma unroll
        for (uint32_t row_i = 0; row_i < block_size_y; row_i++) {
          auto src_reg =
              bias.reg.xetla_select<block_size_x, 1>(j * block_size_x);
          dst_reg.row(row_i) =
              xetla_cvt<dtype_acc, dtype_bias, block_size_x>(src_reg) +
              dst_reg.row(row_i);
        }
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto dst_reg =
            matAcc.reg
                .xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems)
                .xetla_format<dtype_acc, tail_size_y, block_size_x>();
#pragma unroll
        for (uint32_t row_i = 0; row_i < tail_size_y; row_i++) {
          auto src_reg =
              bias.reg.xetla_select<block_size_x, 1>(j * block_size_x);
          dst_reg.row(row_i) =
              xetla_cvt<dtype_acc, dtype_bias, block_size_x>(src_reg) +
              dst_reg.row(row_i);
        }
      }
    }
  }
};

/// @brief Is MatAcc * vector scale + vector offset.
/// @tparam scale_dtype Is the scale data type.
/// @tparam offset_dtype Is the offset data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <
    typename scale_dtype,
    typename offset_dtype,
    gpu_arch arch_tag,
    class enable = void>
struct scale_v_offset_v_op_t {};
/// @brief Is the scale_v_offset_v op functor, specialized for Xe
/// architecture.
template <typename scale_dtype_, typename offset_dtype_, gpu_arch arch_tag>
struct scale_v_offset_v_op_t<
    scale_dtype_,
    offset_dtype_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using scale_dtype = scale_dtype_;
  using offset_dtype = offset_dtype_;

  using scale_mem_desc_t =
      mem_desc_t<scale_dtype, mem_layout::row_major, mem_space::global>;
  using offset_mem_desc_t =
      mem_desc_t<offset_dtype, mem_layout::row_major, mem_space::global>;

  using scale_shape_t = typename scale_mem_desc_t::shape_t;
  using scale_base_t = typename scale_mem_desc_t::base_t;

  using offset_shape_t = typename offset_mem_desc_t::shape_t;
  using offset_base_t = typename offset_mem_desc_t::base_t;

  using coord_t = typename scale_mem_desc_t::coord_t;

  struct arguments_t {
    scale_base_t scale_base;
    scale_shape_t scale_shape;
    offset_base_t offset_base;
    offset_shape_t offset_shape;
    inline arguments_t() = default;
    inline arguments_t(
        scale_base_t scale_base_,
        scale_shape_t scale_shape_,
        offset_base_t offset_base_,
        offset_shape_t offset_shape_)
        : scale_base(scale_base_),
          scale_shape(scale_shape_),
          offset_base(offset_base_),
          offset_shape(offset_shape_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using scale_tile_desc_t =
        tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
    using scale_tile_t = tile_t<scale_dtype, scale_tile_desc_t>;
    using scale_payload_t = mem_payload_t<
        scale_mem_desc_t,
        scale_tile_desc_t,
        msg_type_v<scale_tile_desc_t, scale_mem_desc_t>,
        arch_tag>;
    coord_t scale_coord(coord.x, 0);
    scale_mem_desc_t scale_mem_desc(
        args.scale_base, args.scale_shape, scale_coord);
    scale_tile_t scale_tile;
    scale_payload_t scale_payload(scale_mem_desc);
    tile_load<cache_hint::cached, cache_hint::cached>(
        scale_tile, scale_payload);

    using offset_tile_desc_t =
        tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
    using offset_tile_t = tile_t<offset_dtype, offset_tile_desc_t>;
    using offset_payload_t = mem_payload_t<
        offset_mem_desc_t,
        offset_tile_desc_t,
        msg_type_v<offset_tile_desc_t, offset_mem_desc_t>,
        arch_tag>;
    coord_t offset_coord(coord.x, 0);
    offset_mem_desc_t offset_mem_desc(
        args.offset_base, args.offset_shape, offset_coord);
    offset_tile_t offset_tile;
    offset_payload_t offset_payload(offset_mem_desc);
    tile_load<cache_hint::cached, cache_hint::cached>(
        offset_tile, offset_payload);

#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto acc_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto offset_reg =
            offset_tile.reg.xetla_select<block_size_x, 1>(j * block_size_x);
        auto scale_reg =
            scale_tile.reg.xetla_select<block_size_x, 1>(j * block_size_x);
#pragma unroll
        for (uint32_t row_i = 0; row_i < block_size_y; row_i++) {
          acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x) =
              scale_reg *
                  acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x)

              + offset_reg;
        }
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto acc_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto offset_reg =
            offset_tile.reg.xetla_select<block_size_x, 1>(j * block_size_x);
        auto scale_reg =
            scale_tile.reg.xetla_select<block_size_x, 1>(j * block_size_x);
#pragma unroll
        for (uint32_t row_i = 0; row_i < tail_size_y; row_i++) {
          acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x) =
              scale_reg *
                  acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x) +
              offset_reg;
        }
      }
    }
  }
};

/// @brief Is MatAcc * vector scale.
/// @tparam scale_dtype Is the scale data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename scale_dtype, gpu_arch arch_tag, class enable = void>
struct scale_v_op_t {};
/// @brief Is the scale_v op functor, specialized for Xe architecture.
template <typename scale_dtype_, gpu_arch arch_tag>
struct scale_v_op_t<
    scale_dtype_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using scale_dtype = scale_dtype_;

  using scale_mem_desc_t =
      mem_desc_t<scale_dtype, mem_layout::row_major, mem_space::global>;

  using scale_shape_t = typename scale_mem_desc_t::shape_t;
  using scale_base_t = typename scale_mem_desc_t::base_t;
  using coord_t = typename scale_mem_desc_t::coord_t;

  struct arguments_t {
    scale_base_t scale_base;
    scale_shape_t scale_shape;

    inline arguments_t() = default;
    inline arguments_t(scale_base_t scale_base_, scale_shape_t scale_shape_)
        : scale_base(scale_base_), scale_shape(scale_shape_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using scale_tile_desc_t =
        tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
    using scale_tile_t = tile_t<scale_dtype, scale_tile_desc_t>;
    using scale_payload_t = mem_payload_t<
        scale_mem_desc_t,
        scale_tile_desc_t,
        msg_type_v<scale_tile_desc_t, scale_mem_desc_t>,
        arch_tag>;
    coord_t scale_coord(coord.x, 0);
    scale_mem_desc_t scale_mem_desc(
        args.scale_base, args.scale_shape, scale_coord);
    scale_tile_t scale_tile;
    scale_payload_t scale_payload(scale_mem_desc);
    tile_load<cache_hint::cached, cache_hint::cached>(
        scale_tile, scale_payload);

#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto acc_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto scale_reg =
            scale_tile.reg.xetla_select<block_size_x, 1>(j * block_size_x);
#pragma unroll
        for (uint32_t row_i = 0; row_i < block_size_y; row_i++) {
          acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x) =
              scale_reg *
              acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x);
        }
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto acc_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto scale_reg =
            scale_tile.reg.xetla_select<block_size_x, 1>(j * block_size_x);
#pragma unroll
        for (uint32_t row_i = 0; row_i < tail_size_y; row_i++) {
          acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x) =
              scale_reg *
              acc_reg.xetla_select<block_size_x, 1>(row_i * block_size_x);
        }
      }
    }
  }
};

/// @brief Is the element-wise reduce op functor.
/// Load one buffer from memory and get another from matAcc,
/// element-wise reduce and update the output in place.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam reduce_kind Is the reduce type, can be sum, prod, min and max.
/// @tparam dtype_in Is the memory side buffer data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <
    reduce_op reduce_kind,
    typename dtype_in,
    gpu_arch arch_tag,
    class enable = void>
struct elemwise_reduce_op_t {};
/// @brief Is the element-wise reduce op functor, specialized for Xe
/// architecture.
template <reduce_op reduce_kind_, typename dtype_in_, gpu_arch arch_tag>
struct elemwise_reduce_op_t<
    reduce_kind_,
    dtype_in_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;
  static constexpr reduce_op reduce_kind = reduce_kind_;

  struct arguments_t {
    shape_t shape;
    base_t base;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : shape(shape_), base(base_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using mat_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        block_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        mem_desc_in_t,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t>,
        arch_tag>;
    using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_payload_t mat_in_payload(mem_desc_in);
    mat_in_tile_acc_t mat_in_acc;

#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
      tile_load<cache_hint::cached, cache_hint::cached>(mat_in, mat_in_payload);
      elemwise_cvt(mat_in_acc, mat_in);
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto src_reg =
            mat_in_acc.reg.xetla_select<block_elems, 1>(j * block_elems);
        dst_reg = reduce_helper<reduce_kind, dtype_acc, block_elems>(
            src_reg, dst_reg);
      }
      mat_in_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          block_size_y);
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;

      using mat_tail_in_tile_desc_t = tile_desc_t<
          tile_size_x,
          tail_size_y,
          block_size_x,
          tail_size_y,
          reg_layout::tiled>;
      using mat_tail_in_tile_t = tile_t<dtype_in, mat_tail_in_tile_desc_t>;
      using mat_tail_in_payload_t = mem_payload_t<
          mem_desc_in_t,
          mat_tail_in_tile_desc_t,
          msg_type_v<mat_tail_in_tile_desc_t, mem_desc_in_t>,
          arch_tag>;
      using mat_tail_in_tile_acc_t = tile_t<dtype_acc, mat_tail_in_tile_desc_t>;
      mat_tail_in_tile_t mat_tail_in;
      mat_tail_in_payload_t mat_tail_in_payload(mem_desc_in);
      mat_tail_in_tile_acc_t mat_tail_in_acc;
      mat_tail_in_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          tail_start_y);
      tile_load<cache_hint::cached, cache_hint::cached>(
          mat_tail_in, mat_tail_in_payload);
      elemwise_cvt(mat_tail_in_acc, mat_tail_in);
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_tail_in_acc.reg.xetla_select<tail_block_elems, 1>(
            j * tail_block_elems);
        dst_reg = reduce_helper<reduce_kind, dtype_acc, tail_block_elems>(
            src_reg, dst_reg);
      }
    }
  }
};

/// @brief Is the element-wise reduce op functor, specialized for stream_k
/// dispatch
/// Load partial sum from scratchspace
/// Reduce in GRF
/// Store zero to scratchspace
/// Do these steps with smaller tiles to minimize GRF pressure
/// @tparam reduce_kind Is the reduce type, can be sum, prod, min and max.
/// @tparam dtype_in Is the memory side buffer data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <
    reduce_op reduce_kind,
    typename dtype_in,
    gpu_arch arch_tag,
    class enable = void>
struct elemwise_reduce_op_stream_k_t {};
/// @brief Is the element-wise reduce op functor, specialized for Xe
/// architecture.
template <reduce_op reduce_kind_, typename dtype_in_, gpu_arch arch_tag>
struct elemwise_reduce_op_stream_k_t<
    reduce_kind_,
    dtype_in_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;
  static constexpr reduce_op reduce_kind = reduce_kind_;

  struct arguments_t {
    shape_t shape;
    base_t base;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : shape(shape_), base(base_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      [[maybe_unused]] matAcc_t& matAcc,
      [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using mat_in_tile_desc_t = tile_desc_t<
        block_size_x,
        block_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        mem_desc_in_t,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t>,
        arch_tag>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_tile_t mat_zero(0);
    mat_in_payload_t mat_in_payload(mem_desc_in);

#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        tile_load<cache_hint::cached, cache_hint::cached>(
            mat_in, mat_in_payload);
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);

        auto src_reg = mat_in.reg;
        dst_reg = reduce_helper<reduce_kind, dtype_acc, block_elems>(
            src_reg, dst_reg);

        subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
            mat_zero, mat_in_payload);
        mat_in_payload.template update_tdesc<tdesc_update_dir::x_dir>(
            block_size_x);
      }
      mat_in_payload.template update_tdesc<tdesc_update_dir::x_dir>(
          -num_block_x * block_size_x);
      mat_in_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          block_size_y);
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        tile_load<cache_hint::cached, cache_hint::cached>(
            mat_in, mat_in_payload);
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_in.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        dst_reg = reduce_helper<reduce_kind, dtype_acc, tail_block_elems>(
            src_reg, dst_reg);

        subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
            mat_zero, mat_in_payload);

        mat_in_payload.template update_tdesc<tdesc_update_dir::x_dir>(
            block_size_x);
      }
    }
  }
};

/// @brief Is the dropout op functor.
/// Load the mask from memory and get input from matAcc,
/// do the scaling and zero out, update the output in place.
/// The mask has the same layout as the output.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_mask Is the mask data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_mask, gpu_arch arch_tag, class enable = void>
struct dropout_op_t {};
/// @brief Is the dropout op functor, specialized for Xe architecture.
template <typename dtype_mask_, gpu_arch arch_tag>
struct dropout_op_t<
    dtype_mask_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_mask = dtype_mask_;
  using mem_desc_mask_t =
      mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_mask_t::shape_t;
  using coord_t = typename mem_desc_mask_t::coord_t;
  using base_t = typename mem_desc_mask_t::base_t;
  static constexpr uint32_t num_flag = 4;
  static constexpr uint32_t unroll_size = num_flag * 16;
  struct arguments_t {
    shape_t shape;
    base_t base;
    float prob;
    float scale;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_, float prob_, float scale_)
        : shape(shape_), base(base_), prob(prob_), scale(scale_) {}
  };

  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    if (args.prob == 0) {
      return;
    }
    using mask_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mask_in_tile_t = tile_t<dtype_mask, mask_in_tile_desc_t>;
    using mask_in_payload_t = mem_payload_t<
        mem_desc_mask_t,
        mask_in_tile_desc_t,
        msg_type_v<mask_in_tile_desc_t, mem_desc_mask_t>,
        arch_tag>;
    mem_desc_mask_t mem_desc_mask(args.base, args.shape, coord);
    mask_in_tile_t mask_in;
    mask_in_payload_t mask_in_payload(mem_desc_mask);
    tile_load<cache_hint::cached, cache_hint::cached>(mask_in, mask_in_payload);
    if constexpr (tile_elems / unroll_size != 0) {
#pragma unroll
      for (uint32_t i = 0; i < tile_elems / unroll_size; i++) {
        xetla_mask<unroll_size> mask_flag =
            mask_in.reg.xetla_select<unroll_size, 1>(i * unroll_size) > 0;
        auto dst_reg = matAcc.reg.xetla_select<unroll_size, 1>(i * unroll_size);
        dst_reg *= args.scale;
        dst_reg.xetla_merge(0, mask_flag);
      }
    }
    if constexpr (tile_elems % unroll_size != 0) {
      constexpr uint32_t remain_len = tile_elems % unroll_size;
      constexpr uint32_t remain_start = tile_elems / unroll_size * unroll_size;
      xetla_mask<remain_len> mask_flag =
          mask_in.reg.xetla_select<remain_len, 1>(remain_start) > 0;
      auto dst_reg = matAcc.reg.xetla_select<remain_len, 1>(remain_start);
      dst_reg *= args.scale;
      dst_reg.xetla_merge(0, mask_flag);
    }
  }
};

/// @brief Is the random number generator and dropout op functor.
/// Generate the mask data and get input from matAcc, do the scaling and zero
/// out, update the output in place, dump the mask buffer to memory. The mask
/// has the same layout as the output. Used in epilogue::tile_op or
/// chained_tile_op.
/// @tparam dtype_mask Is the mask data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_mask, gpu_arch arch_tag, class enable = void>
struct rng_dropout_op_t {};
/// @brief Is the random number generator and dropout op functor, specialized
/// for Xe architecture.
template <typename dtype_mask_, gpu_arch arch_tag>
struct rng_dropout_op_t<
    dtype_mask_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_mask = dtype_mask_;
  using mem_desc_mask_t =
      mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_mask_t::shape_t;
  using coord_t = typename mem_desc_mask_t::coord_t;
  using base_t = typename mem_desc_mask_t::base_t;
  static constexpr uint32_t random_simd = 16;
  static constexpr uint32_t random_len = 4 * random_simd;
  xetla_rand_t<random_simd> rand_gen;

  struct arguments_t {
    shape_t mask_shape;
    base_t mask_base;
    uint64_t* rand_offset_ptr;
    float prob;
    uint64_t rand_seed;

    inline arguments_t() = default;
    inline arguments_t(
        base_t mask_base_,
        shape_t mask_shape_,
        float prob_,
        uint64_t* rand_offset_ptr_,
        uint64_t rand_seed_ = 67280421310721)
        : mask_shape(mask_shape_),
          mask_base(mask_base_),
          rand_offset_ptr(rand_offset_ptr_),
          prob(prob_),
          rand_seed(rand_seed_) {}
  };

  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;

    using mask_out_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mask_out_tile_t = tile_t<dtype_mask, mask_out_tile_desc_t>;
    using mask_out_payload_t = mem_payload_t<
        mem_desc_mask_t,
        mask_out_tile_desc_t,
        msg_type_v<mask_out_tile_desc_t, mem_desc_mask_t>,
        arch_tag>;
    if (args.prob == 0) {
      return;
    }
    // calculate the scale internally
    float scale = 1.f / (1.f - args.prob);
    uint32_t threshold = uint32_t(args.prob * float(4294967296));
    xetla_vector<uint64_t, 1> rand_offset_v =
        xetla_load_global<uint64_t, 1, cache_hint::cached, cache_hint::cached>(
            args.rand_offset_ptr, 0);
    uint64_t rand_offset = rand_offset_v[0];
    uint64_t rand_subseq = uint64_t(coord.y) << 32 | uint64_t(coord.x);
    rand_gen.init(args.rand_seed, rand_subseq, rand_offset);

    mem_desc_mask_t mem_desc_mask(args.mask_base, args.mask_shape, coord);
    mask_out_tile_t mask_out;
    mask_out_payload_t mask_out_payload(mem_desc_mask);
    if constexpr (tile_elems / random_len != 0) {
#pragma unroll
      for (uint32_t i = 0; i < tile_elems / random_len; i++) {
        auto out_sub = matAcc.reg.xetla_select<random_len, 1>(i * random_len);
        auto mask_sub =
            mask_out.reg.xetla_select<random_len, 1>(i * random_len);
        xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
        xetla_mask<random_len> mask_flag = rand_val < threshold;
        out_sub *= scale;
        out_sub.xetla_merge(0, mask_flag);
        mask_sub.xetla_merge(1, 0, mask_flag);
      }
    }
    if constexpr (tile_elems % random_len != 0) {
      constexpr uint32_t remain_len = tile_elems % random_len;
      constexpr uint32_t remain_start = tile_elems / random_len * random_len;
      auto out_sub = matAcc.reg.xetla_select<remain_len, 1>(remain_start);
      auto mask_sub = mask_out.reg.xetla_select<remain_len, 1>(remain_start);
      // dropout, still generate random_len
      xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
      xetla_mask<random_len> mask_flag = rand_val < threshold;
      out_sub *= scale;
      out_sub.xetla_merge(0, mask_flag.xetla_select<remain_len, 1>(0));
      mask_sub.xetla_merge(1, 0, mask_flag.xetla_select<remain_len, 1>(0));
    }
    tile_store<cache_hint::streaming>(mask_out, mask_out_payload);
  }
};

/// @brief Is the scalar_multiply op functor.
/// Get the input from matAcc, multiply with a scalar, update the output in
/// place. Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_in Is the data type of multiplier buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_in, gpu_arch arch_tag, class enable = void>
struct scalar_mul_op_t {};
/// @brief Is the scalar_multiply op functor, specialized for Xe architecture.
template <typename dtype_in_, gpu_arch arch_tag>
struct scalar_mul_op_t<
    dtype_in_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using coord_t = typename mem_desc_in_t::coord_t;

  struct arguments_t {
    dtype_in multiplier;
    inline arguments_t() = default;
    inline arguments_t(dtype_in multiplier_) : multiplier(multiplier_) {}
  };

  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      [[maybe_unused]] coord_t coord,
      [[maybe_unused]] arguments_t args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static_assert(
        std::is_same<dtype_in, dtype_acc>::value,
        "Given multiplier must have same type as matAcc!");
    matAcc.reg *= args.multiplier;
  }
};

/// @brief Is the linear_op functor.
/// Multiply matAcc with a scalar, then add a bias, update the output in
/// place. Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_in Is the data type of multiplier buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_in, gpu_arch arch_tag, class enable = void>
struct linear_op_t {};
/// @brief Is the linear_op functor, specialized for Xe architecture.
template <typename dtype_in_, gpu_arch arch_tag>
struct linear_op_t<
    dtype_in_,
    arch_tag,
    std::enable_if_t<valid_xe_arch_tag<arch_tag>>> {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    dtype_in alpha;
    dtype_in beta;
    inline arguments_t() = default;
    inline arguments_t(
        base_t base_,
        shape_t shape_,
        dtype_in alpha_,
        dtype_in beta_)
        : shape(shape_), base(base_), alpha(alpha_), beta(beta_) {}
  };

  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr uint32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t num_block_y = matAcc_t::num_block_y;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;
    static constexpr uint32_t remained_size_y = tile_size_y % block_size_y;

    using mat_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        block_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        mem_desc_in_t,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t>,
        arch_tag>;
    using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_payload_t mat_in_payload(mem_desc_in);
    mat_in_tile_acc_t mat_in_acc;

    dtype_acc alpha = dtype_acc(args.alpha);
    dtype_acc beta = dtype_acc(args.beta);
    matAcc.reg *= alpha;

#pragma unroll
    for (uint32_t i = 0; i < num_block_y; ++i) {
      tile_load<cache_hint::cached, cache_hint::cached>(mat_in, mat_in_payload);
      elemwise_cvt(mat_in_acc, mat_in);
      mat_in_acc.reg *= beta;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto src_reg =
            mat_in_acc.reg.xetla_select<block_elems, 1>(j * block_elems);
        dst_reg = reduce_helper<reduce_op::sum, dtype_acc, block_elems>(
            src_reg, dst_reg);
      }
      mat_in_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          block_size_y);
    }
    // process the tail
    if constexpr (remained_size_y > 0) {
      constexpr uint32_t tail_start_y = num_block_y * block_size_y;
      constexpr uint32_t tail_block_elems = remained_size_y * block_size_x;

      using mat_tail_in_tile_desc_t = tile_desc_t<
          tile_size_x,
          remained_size_y,
          block_size_x,
          remained_size_y,
          reg_layout::tiled>;
      using mat_tail_in_tile_t = tile_t<dtype_in, mat_tail_in_tile_desc_t>;
      using mat_tail_in_payload_t = mem_payload_t<
          mem_desc_in_t,
          mat_tail_in_tile_desc_t,
          msg_type_v<mat_tail_in_tile_desc_t, mem_desc_in_t>,
          arch_tag>;
      using mat_tail_in_tile_acc_t = tile_t<dtype_acc, mat_tail_in_tile_desc_t>;

      mat_tail_in_tile_t mat_tail_in;
      mat_tail_in_payload_t mat_tail_in_payload(mem_desc_in);
      mat_tail_in_tile_acc_t mat_tail_in_acc;
      mat_tail_in_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          tail_start_y);
      tile_load<cache_hint::cached, cache_hint::cached>(
          mat_tail_in, mat_tail_in_payload);
      elemwise_cvt(mat_tail_in_acc, mat_tail_in);
      mat_tail_in_acc.reg *= beta;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_tail_in_acc.reg.xetla_select<tail_block_elems, 1>(
            j * tail_block_elems);
        dst_reg = reduce_helper<reduce_op::sum, dtype_acc, tail_block_elems>(
            src_reg, dst_reg);
      }
    }
  }
};

} // namespace gpu::xetla::subgroup
