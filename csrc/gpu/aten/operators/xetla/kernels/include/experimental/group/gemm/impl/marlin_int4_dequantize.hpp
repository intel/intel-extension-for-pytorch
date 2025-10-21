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

#define UsePreShuffle 1

#pragma once

#include <experimental/group/gemm/common.hpp>
#include <experimental/group/gemm/compute_policy.hpp>
#include "prefetch.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief Is the gemm functor for Xe architecture and matrix engine.
template <
    DequantMode dequant_mode_,
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename tile_shape_,
    typename mem_desc_a_t_,
    typename mem_desc_b_t_,
    typename dtype_scale_,
    int dequant_s_,
    typename pre_processing_t_,
    gpu_arch arch_tag_>
class gemm_t<
    compute_policy_int4_dequantize_v2<
        compute_attr_,
        perf_tuning_knob_,
        dtype_scale_,
        dequant_s_,
        dequant_mode_,
        arch_tag_>,
    tile_shape_, // tile shape of workgroup-level gemm
    mem_desc_a_t_, // memory attribute of matA
    mem_desc_b_t_, // memory attribute of matB
    pre_processing_t_, // pre_processing functor
    std::enable_if_t<(arch_tag_ == gpu_arch::XeHpc)>> {
 public:
  using mem_desc_a_t = mem_desc_a_t_;
  using mem_desc_b_t = mem_desc_b_t_;
  using tile_shape = tile_shape_;
  using pre_processing_t = pre_processing_t_;
  using compute_policy = compute_policy_int4_dequantize_v2<
      compute_attr_,
      perf_tuning_knob_,
      dtype_scale_,
      dequant_s_,
      dequant_mode_,
      arch_tag_>;
  static constexpr uint32_t k_stride = compute_policy::k_stride;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  using work_group_t = typename tile_shape::work_group_t;

  constexpr static gpu_arch arch_tag = compute_policy::arch_tag;

  static constexpr mem_layout mem_layout_a = mem_desc_a_t::layout;
  static constexpr mem_layout mem_layout_b = mem_desc_b_t::layout;
  static constexpr bool is_col_major_a = mem_layout_a == mem_layout::col_major;
  static constexpr bool is_col_major_b = mem_layout_b == mem_layout::col_major;
  using dtype_a = typename mem_desc_a_t::dtype;
  using dtype_b = typename mem_desc_b_t::dtype;
  // pack ratio for int4
  static constexpr uint32_t pack_ratio = sizeof(dtype_b) * 2;
  static_assert(is_col_major_b == false, "Only support row-major for matB");
  static_assert(pack_ratio == 8, "Only support pack ratio 8 for int4");
  static_assert(sizeof(dtype_b) == 4, "Only support int32_t/uint32_t for matB");

  // below are constexpr used for dequant
  // note: we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  static constexpr int LO = 0x000f000f;
  static constexpr int HI = 0x00f000f0;
  static constexpr int EX = 0x64006400;
  static constexpr int SUB = 1024 + 8;
  static constexpr int DIV = 16;
  static constexpr int ADD = 72;

 private:
  /******** set data type **********/
  using dtype_mma_acc = typename compute_policy::dtype_mma_acc;
  using dtype_mma_a = typename compute_policy::dtype_mma_a;
  using dtype_mma_b = typename compute_policy::dtype_mma_b;

  // check data type
  static_assert(
      (sizeof(dtype_mma_a) == sizeof(dtype_a)) ||
          (sizeof(dtype_mma_a) == 2 * sizeof(dtype_a)) ||
          (2 * sizeof(dtype_mma_a) == sizeof(dtype_a)),
      "Current we cannot support fp32 <->fp8, since it will "
      "meet a lot of HW limitations. ");

  /******** set memory attribute **********/
  static constexpr mem_space mem_space_a = mem_desc_a_t::space;
  static constexpr mem_space mem_space_b = mem_desc_b_t::space;

  static constexpr bool is_local_a = mem_space_a == mem_space::local;
  static constexpr bool is_local_b = mem_space_b == mem_space::local;
  static constexpr tdesc_update_dir update_dir_a =
      is_col_major_a ? tdesc_update_dir::y_dir : tdesc_update_dir::x_dir;
  static constexpr tdesc_update_dir update_dir_b =
      is_col_major_b ? tdesc_update_dir::x_dir : tdesc_update_dir::y_dir;

  // check memory type
  static_assert(!is_col_major_b, "only support MatB row-major for now");
  static_assert(
      (!is_local_a) && (!is_local_b),
      "only support from global memory for now");

  static constexpr uint32_t stages = compute_policy::stages;
  static constexpr uint32_t sync_freq = compute_policy::sync_freq;

  /******** set tile layout && worker scope **********/
  static constexpr uint32_t tile_size_x_a = k_stride;
  static constexpr uint32_t tile_size_y_a = sg_tile_m;
  static constexpr uint32_t tile_size_x_b = sg_tile_n;
  static constexpr uint32_t tile_size_y_b = k_stride;
  static constexpr uint32_t tile_size_x_c = sg_tile_n;
  static constexpr uint32_t tile_size_y_c = sg_tile_m;
  static constexpr uint32_t block_size_x_a = compute_policy::block_size_x_a;
  static constexpr uint32_t block_size_y_a =
      (compute_policy::block_size_y_a > tile_size_y_a)
      ? tile_size_y_a
      : compute_policy::block_size_y_a;
  static constexpr uint32_t block_size_x_b = compute_policy::block_size_x_b;
  static constexpr uint32_t block_size_y_b = compute_policy::block_size_y_b;

  /******** set tile  **********/
  // for matA
  static constexpr bool is_vnni_tiled_a =
      (sizeof(dtype_a) < sizeof(uint32_t)) && is_col_major_a;
  static constexpr reg_layout reg_layout_a =
      is_vnni_tiled_a ? reg_layout::vnni_tiled : reg_layout::tiled;
  using matA_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_a,
      tile_size_y_a,
      block_size_x_a,
      block_size_y_a,
      reg_layout_a>;
  using matA_t = subgroup::tile_t<dtype_a, matA_tile_desc_t>;
  using matA_payload_t = subgroup::mem_payload_t<
      mem_desc_a_t,
      matA_tile_desc_t,
      subgroup::msg_type_v<matA_tile_desc_t, mem_desc_a_t>,
      arch_tag>;
  using matA_acc_t = subgroup::tile_t<dtype_mma_a, matA_tile_desc_t>;
  using matA_prefetch_payload_t = subgroup::prefetch_payload_t<
      mem_desc_a_t,
      subgroup::tile_desc_t<tile_size_x_a, tile_size_y_a, 1, 1>,
      wg_size_x,
      arch_tag>;

  // for matB
  using matB_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_b,
      tile_size_y_b / pack_ratio,
      block_size_x_b,
      block_size_y_b / pack_ratio,
      reg_layout::tiled>;
  using matB_t = subgroup::tile_t<dtype_b, matB_tile_desc_t>;
  using matB_payload_t = subgroup::mem_payload_t<
      mem_desc_b_t,
      matB_tile_desc_t,
      subgroup::msg_type_v<matB_tile_desc_t, mem_desc_b_t>,
      arch_tag>;
  using matB_prefetch_payload_t = subgroup::
      prefetch_payload_t<mem_desc_b_t, matB_tile_desc_t, wg_size_y, arch_tag>;

  using matB_acc_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_b,
      tile_size_y_b,
      block_size_x_b,
      block_size_y_b,
      reg_layout::vnni_tiled>;
  using matB_acc_t = subgroup::tile_t<dtype_mma_b, matB_acc_tile_desc_t>;

 public:
  static constexpr uint32_t dequant_s = compute_policy::dequant_s;
  using dtype_scale = typename compute_policy::dtype_scale;
  using mem_desc_scale_t =
      mem_desc_t<dtype_scale, mem_layout::row_major, mem_space::global>;

 private:
  /******** set tile layout for scale **********/
  static_assert(
      (k_stride % block_size_y_b == 0),
      "k_stride%(block_size_y_b) == 0");
  static_assert(
      (dequant_s % block_size_y_b == 0),
      "dequant_s%(block_size_y_b) == 0");
  static_assert(
      (dequant_s % k_stride == 0),
      "dequant_s is expected to be a multiple of k_stride(16, 32)");

  // for scale, num_block_y set to 1
  static constexpr uint32_t block_size_y_scale =
      (k_stride + dequant_s - 1) / dequant_s;
  static constexpr uint32_t tile_size_y_scale = block_size_y_scale;

  static constexpr uint32_t scale_addr_update_freq = dequant_s / k_stride;

  using scale_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_b,
      tile_size_y_scale,
      block_size_x_b,
      block_size_y_scale,
      reg_layout::tiled>;

  using scale_t = subgroup::tile_t<dtype_scale, scale_tile_desc_t>;
  using scale_payload_t = subgroup::mem_payload_t<
      mem_desc_scale_t,
      scale_tile_desc_t,
      subgroup::msg_type_v<scale_tile_desc_t, mem_desc_scale_t>,
      arch_tag>;
  using scale_prefetch_payload_t = subgroup::block_prefetch_payload_t<
      mem_desc_scale_t,
      scale_tile_desc_t,
      1,
      arch_tag>;

 public:
  using matAcc_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_c,
      tile_size_y_c,
      block_size_x_b,
      block_size_y_a,
      reg_layout::tiled>;
  using matAcc_t = subgroup::tile_t<dtype_mma_acc, matAcc_tile_desc_t>;

 private:
  using tile_mma = subgroup::tile_mma_t<
      matAcc_t,
      matAcc_t,
      matB_acc_t,
      matA_acc_t,
      mma_engine::xmx,
      arch_tag>;
  static constexpr bool enable_periodic_sync = (sync_freq != 0);
  static constexpr uint32_t barrier_count_x = wg_size_y > 1 ? wg_size_x : 0;
  static constexpr uint32_t barrier_count_y = wg_size_x > 1 ? wg_size_y : 0;

 public:
  static constexpr uint32_t barrier_count =
      enable_periodic_sync ? barrier_count_x + barrier_count_y : 0;

  static constexpr uint32_t slm_size = 0;

  static constexpr msg_type msg_type_a = matA_payload_t::message_type;
  static constexpr msg_type msg_type_b = matB_payload_t::message_type;

  /// @brief Arguments for gemm.
  /// User should prepare matA_base_desc, matB_base_desc, inner_loop_count...
  struct arguments_t {
    /// @brief Is the memory description of matA, including base, shape and
    /// coordinate.
    mem_desc_a_t matA_base_desc;
    /// @brief Is the memory description of matB, including base, shape and
    /// coordinate.
    mem_desc_b_t matB_base_desc;
    /// @brief Is the total inner loop count required to compute the entire
    /// K-dim.
    uint32_t inner_loop_count;
    /// @brief Is the memory description of scale buffer.
    /// Scale size: (matrix_k/dequant_s)x(matrix_n)
    mem_desc_scale_t scale_base_desc;

    /// @brief Default construct.
    inline arguments_t() = default;

    inline arguments_t(
        mem_desc_a_t matA_desc,
        mem_desc_b_t matB_desc,
        uint32_t loop_count,
        mem_desc_scale_t scale_desc)
        : matA_base_desc(matA_desc),
          matB_base_desc(matB_desc),
          inner_loop_count(loop_count),
          scale_base_desc(scale_desc) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment,
    // destructor) Please check if you need to add self-define destructor inline
    // ~arguments_t(){}
    inline arguments_t(const arguments_t& args)
        : matA_base_desc(args.matA_base_desc),
          matB_base_desc(args.matB_base_desc),
          inner_loop_count(args.inner_loop_count),
          scale_base_desc(args.scale_base_desc) {}
    inline arguments_t& operator=(const arguments_t& args) {
      this->matA_base_desc = args.matA_base_desc;
      this->matB_base_desc = args.matB_base_desc;
      this->inner_loop_count = args.inner_loop_count;
      this->scale_base_desc = args.scale_base_desc;
      return *this;
    }
    inline void init(
        mem_desc_a_t matA_desc,
        mem_desc_b_t matB_desc,
        uint32_t loop_count,
        mem_desc_scale_t scale_desc) {
      matA_base_desc = matA_desc;
      matB_base_desc = matB_desc;
      inner_loop_count = loop_count;
      scale_base_desc = scale_desc;
    }
  };

  /// @brief Main execution function for gemm.
  /// The basic process is load data -> matrix multiply.
  /// @param g Is the workgroup of the current tile.
  /// @param matAcc Is the reference of the accumulation buffer.
  /// @param args Is the gemm::arguments_t.
  /// @param slm_base Is the slm base address.
  /// @param nbarrier_base Is the named barrier base.
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g,
      matAcc_t& matAcc,
      arguments_t args,
      [[maybe_unused]] uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    int32_t sg_idx = g.get_id() % wg_size_x;
    int32_t sg_idy = g.get_id() / wg_size_x;
    update_sg_tile_tdesc(args, sg_idx, sg_idy);

    matA_t matA;
    matB_t matB;
    scale_t scale;

    matA_payload_t matA_payload(args.matA_base_desc);
    matB_payload_t matB_payload(args.matB_base_desc);
    scale_payload_t scale_payload(args.scale_base_desc);
    matA_prefetch_payload_t matA_prefetch_payload(args.matA_base_desc, sg_idx);
    matB_prefetch_payload_t matB_prefetch_payload(args.matB_base_desc, sg_idy);
    scale_prefetch_payload_t scale_prefetch_payload(args.scale_base_desc, 0);

    xetla_nbarrier_t<wg_size_x, wg_size_x, arch_tag> nbarrier_a;
    nbarrier_a.init_nbarrier(
        sg_idy + nbarrier_base, nbarrier_role::producer_consumer);
    xetla_nbarrier_t<wg_size_y, wg_size_y, arch_tag> nbarrier_b;
    nbarrier_b.init_nbarrier(
        sg_idx + barrier_count_y + nbarrier_base,
        nbarrier_role::producer_consumer);

    SW_BARRIER();
    // prefetch A and B and scale
#pragma unroll
    for (uint32_t i = 0; i < stages; i++) {
      subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          matA_prefetch_payload);
      subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          matB_prefetch_payload);

      matA_prefetch_payload.template update_tdesc<update_dir_a>(
          matA_t::tile_size_x);
      matB_prefetch_payload.template update_tdesc<update_dir_b>(
          matB_t::tile_size_y);
      if ((i % scale_addr_update_freq) == 0) {
        subgroup::block_tile_prefetch<cache_hint::cached, cache_hint::cached>(
            scale_prefetch_payload);
        scale_prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(
            scale_t::tile_size_y);
      }
    }

    for (uint32_t i = 0; i < args.inner_loop_count; i++) {
      if constexpr (enable_periodic_sync) {
        if ((i % sync_freq) == 0) {
          if constexpr (wg_size_x > 1) {
            nbarrier_a.arrive();
          }
          if constexpr (wg_size_y > 1) {
            nbarrier_b.arrive();
          }
        }
      }
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matA, matA_payload);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matB, matB_payload);

      SW_BARRIER();
      if constexpr (stages != 0) {
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            matA_prefetch_payload);
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            matB_prefetch_payload);
        if ((i % scale_addr_update_freq) == 0) {
          subgroup::block_tile_prefetch<cache_hint::cached, cache_hint::cached>(
              scale_prefetch_payload);
        }
      }
      SW_BARRIER();
      matA_payload.template update_tdesc<update_dir_a>(matA_t::tile_size_x);
      matB_payload.template update_tdesc<update_dir_b>(matB_t::tile_size_y);
      if ((i % scale_addr_update_freq) == 0) {
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
            scale, scale_payload);
        scale_payload.template update_tdesc<tdesc_update_dir::y_dir>(
            scale_t::tile_size_y);
      }
      if constexpr (stages != 0) {
        matA_prefetch_payload.template update_tdesc<update_dir_a>(
            matA_t::tile_size_x);
        matB_prefetch_payload.template update_tdesc<update_dir_b>(
            matB_t::tile_size_y);
        if ((i % scale_addr_update_freq) == 0) {
          scale_prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(
              scale_t::tile_size_y);
        }
      }
      SW_BARRIER();
      matA_acc_t matA_acc;
      matB_acc_t matB_acc;
      if constexpr (is_vnni_tiled_a) {
        subgroup::vnni_reverse(matA);
      }
      subgroup::elemwise_cvt(matA_acc, matA);
      dequantize(i, matB_acc, matB, scale);
      SW_BARRIER();
      tile_mma::mma(matAcc, matAcc, matB_acc, matA_acc);
      SW_BARRIER();
      if constexpr (enable_periodic_sync) {
        if ((i % sync_freq) == 0) {
          if constexpr (wg_size_x > 1) {
            nbarrier_a.wait();
          }
          if constexpr (wg_size_y > 1) {
            nbarrier_b.wait();
          }
        }
      }
    }
    SW_BARRIER();
  }

 private:
  template <DequantMode dm = dequant_mode_>
  inline std::enable_if_t<dm == DequantMode::Basic, void> dequantize(
      int step,
      matB_acc_t& matB_acc,
      matB_t& matB,
      scale_t& scale) {
    // no tail, because this is matB
    constexpr uint32_t num_block_x = tile_size_x_b / block_size_x_b;
    constexpr uint32_t num_block_y = tile_size_y_b / block_size_y_b;
    constexpr uint32_t vnni_rows = sizeof(uint32_t) / sizeof(dtype_mma_b);
    constexpr uint32_t block_b_y_per_scale = dequant_s / block_size_y_b;
#pragma unroll
    for (uint32_t i = 0; i < num_block_y; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        int block_id = (i * num_block_x + j);
        auto matB_blk = matB.reg
                            .xetla_select<matB_t::block_elems, 1>(
                                block_id * matB_t::block_elems)
                            .xetla_format<uint8_t>();

        xetla_vector<int8_t, matB_acc_t::block_elems> cvt_blk;
        cvt_blk.xetla_select<matB_t::block_elems * 4, 2>(0) = matB_blk & 0x0f;
        cvt_blk.xetla_select<matB_t::block_elems * 4, 2>(1) =
            (matB_blk >> 4) & 0x0f;
        // zero point equal to 8
        cvt_blk = cvt_blk - int8_t(8);

        int scale_block_id = (i / block_b_y_per_scale * num_block_x + j);
        auto scale_vec = scale.reg.xetla_select<scale_t::block_size_x, 1>(
            scale_block_id * scale_t::block_size_x);

        auto dst_blk = matB_acc.reg.xetla_select<matB_acc_t::block_elems, 1>(
            block_id * matB_acc_t::block_elems);
#pragma unroll
        for (uint32_t k = 0; k < block_size_y_b / pack_ratio; ++k) {
#pragma unroll
          for (uint32_t l = 0; l < pack_ratio / vnni_rows; ++l) {
#pragma unroll
            for (uint32_t m = 0; m < vnni_rows; ++m) {
              uint32_t offset = k * block_size_x_b * pack_ratio;
              auto dst = dst_blk.xetla_select<block_size_x_b, vnni_rows>(
                  offset + l * block_size_x_b * vnni_rows + m);
              // convert int8 to fp16
              dst = cvt_blk.xetla_select<block_size_x_b, pack_ratio>(
                  offset + l * vnni_rows + m);
              dst = dst * scale_vec;
            }
          }
        }
        // if (step == 0 && i == 0 && j == 0) {
        //         xetla_vector<float, matB_acc_t::block_elems> dst_blk_f =
        //         dst_blk; dump<float, matB_acc_t::block_elems>(dst_blk_f,
        //         block_size_y_b, block_size_x_b);
        // }
      }
    }
  }

  template <DequantMode dm = dequant_mode_>
  inline std::enable_if_t<dm == DequantMode::FastInterleaved, void> dequantize(
      int step,
      matB_acc_t& matB_acc,
      matB_t& matB,
      scale_t& scale) {
    // no tail, because this is matB
    constexpr uint32_t num_block_x = tile_size_x_b / block_size_x_b;
    constexpr uint32_t num_block_y = tile_size_y_b / block_size_y_b;
    constexpr uint32_t packed_block_size_y = block_size_y_b / pack_ratio;
    constexpr uint32_t vnni_rows = sizeof(uint32_t) / sizeof(dtype_mma_b);
    constexpr uint32_t block_b_y_per_scale = dequant_s / block_size_y_b;

    // Efficiently dequantize an int32 value into 4 fp16 values.
    static_assert(
        pack_ratio % 8 == 0, "pack ratio is expected to be a multiple of 8");
#pragma unroll
    for (uint32_t i = 0; i < num_block_y; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        int block_id = (i * num_block_x + j);

        // view as u8 (01, 23, 45, 67)
        // shape: (block_size_y_b, block_size_x_b, 4)
        auto matB_blk = matB.reg
                            .xetla_select<matB_t::block_elems, 1>(
                                block_id * matB_t::block_elems)
                            .xetla_format<uint8_t, matB_t::block_elems, 4>();
        // view as u16
        // shape: (block_size_y_b * 2, block_size_x_b * 2)
        // 01 | 23
        // 45 | 67
        xetla_vector<uint16_t, matB_t::block_elems * 4> u16_matB_blk;
        for (uint32_t k = 0; k < packed_block_size_y; ++k) {
          u16_matB_blk.template select<2 * block_size_x_b, 1>(
              k * 4 * block_size_x_b) =
              matB_blk.template select<block_size_x_b, 1, 2, 1>(
                  k * block_size_x_b, 0);
          u16_matB_blk.template select<2 * block_size_x_b, 1>(
              (k * 4 + 2) * block_size_x_b) =
              matB_blk.template select<block_size_x_b, 1, 2, 1>(
                  k * block_size_x_b, 2);
        }
        // view as u32 for fast Interleaved dequant
        xetla_vector<uint32_t, matB_t::block_elems* 2> u32_matB_blk =
            u16_matB_blk.xetla_format<uint32_t>();

        // directly transitioned from integer encoding to floating-point
        // encoding. lo:
        //  02 | 02 | ... | 02
        //  46 | 46 | ... | 46
        //  810 | 810 | ... | 810
        // 1214| 1214| ... | 1214
        // hi:
        //  13 | 13 | ... | 13
        //  57 | 57 | ... | 57
        //  911 | 911 | ... | 911
        //  1315| 1315| ... | 1315
        xetla_vector<uint32_t, matB_t::block_elems* 2> u32_hi_matB =
            (u32_matB_blk & HI) | EX;
        xetla_vector<uint32_t, matB_t::block_elems* 2> u32_lo_matB =
            (u32_matB_blk & LO) | EX;

        auto f16_hi_matB = u32_hi_matB.xetla_format<sycl::half>();
        auto f16_lo_matB = u32_lo_matB.xetla_format<sycl::half>();

        f16_lo_matB -= SUB;
        f16_hi_matB /= DIV;
        f16_hi_matB -= ADD;

        int scale_block_id = (i / block_b_y_per_scale * num_block_x + j);
        auto scale_vec = scale.reg.xetla_select<scale_t::block_size_x, 1>(
            scale_block_id * scale_t::block_size_x);
        xetla_vector<dtype_scale, scale_t::block_size_x * vnni_rows>
            scale_vec_vnni;
        for (uint32_t k = 0; k < vnni_rows; ++k) {
          scale_vec_vnni.template select<scale_t::block_size_x, vnni_rows>(k) =
              scale_vec;
        }

        auto dst_blk = matB_acc.reg.xetla_select<matB_acc_t::block_elems, 1>(
            block_id * matB_acc_t::block_elems);
#pragma unroll
        for (uint32_t k = 0; k < packed_block_size_y; ++k) {
          uint32_t dst_offset = k * block_size_x_b * pack_ratio;
          uint32_t src_offset = k * block_size_x_b * 4;
          auto lo_dst =
              dst_blk.template select<4 * block_size_x_b, 1>(dst_offset);
          auto hi_dst = dst_blk.template select<4 * block_size_x_b, 1>(
              dst_offset + 4 * block_size_x_b);
          lo_dst =
              f16_lo_matB.template select<4 * block_size_x_b, 1>(src_offset);
          hi_dst =
              f16_hi_matB.template select<4 * block_size_x_b, 1>(src_offset);
        }

        for (uint32_t k = 0; k < block_size_y_b / vnni_rows; k++) {
          auto dst = dst_blk.template select<block_size_x_b * vnni_rows, 1>(
              k * block_size_x_b * vnni_rows);
          dst *= scale_vec_vnni;
        }
      }
    }
  }

  template <DequantMode dm = dequant_mode_>
  inline std::
      enable_if_t<dm == DequantMode::FastInterleavedWithScaleMerge, void>
      dequantize(int step, matB_acc_t& matB_acc, matB_t& matB, scale_t& scale) {
    // no tail, because this is matB
    constexpr uint32_t num_block_x = tile_size_x_b / block_size_x_b;
    constexpr uint32_t num_block_y = tile_size_y_b / block_size_y_b;
    constexpr uint32_t packed_block_size_y = block_size_y_b / pack_ratio;
    constexpr uint32_t vnni_rows = sizeof(uint32_t) / sizeof(dtype_mma_b);
    constexpr uint32_t block_b_y_per_scale = dequant_s / block_size_y_b;

    // Efficiently dequantize an int32 value into 4 fp16 values.
    constexpr uint32_t parallel_dequantize_int4_count = 4;
    static_assert(
        pack_ratio % parallel_dequantize_int4_count == 0,
        "pack ratio is expected to be a multiple of 4");
#pragma unroll
    for (uint32_t i = 0; i < num_block_y; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; ++j) {
        int block_id = (i * num_block_x + j);

        auto matB_blk = matB.reg
                            .xetla_select<matB_t::block_elems, 1>(
                                block_id * matB_t::block_elems)
                            .xetla_format<uint8_t, matB_t::block_elems, 4>();

        xetla_vector<uint16_t, matB_t::block_elems * 4> u16_matB_blk;
        for (uint32_t k = 0; k < packed_block_size_y; ++k) {
          u16_matB_blk.template select<2 * block_size_x_b, 1>(
              k * 4 * block_size_x_b) =
              matB_blk.template select<block_size_x_b, 1, 2, 1>(
                  k * block_size_x_b, 0);
          u16_matB_blk.template select<2 * block_size_x_b, 1>(
              (k * 4 + 2) * block_size_x_b) =
              matB_blk.template select<block_size_x_b, 1, 2, 1>(
                  k * block_size_x_b, 2);
        }
        xetla_vector<uint32_t, matB_t::block_elems* 2> u32_matB_blk =
            u16_matB_blk.xetla_format<uint32_t>();

        xetla_vector<uint32_t, matB_t::block_elems* 2> u32_hi_matB =
            (u32_matB_blk & HI) | EX;
        xetla_vector<uint32_t, matB_t::block_elems* 2> u32_lo_matB =
            (u32_matB_blk & LO) | EX;

        auto f16_hi_matB = u32_hi_matB.xetla_format<sycl::half>();
        auto f16_lo_matB = u32_lo_matB.xetla_format<sycl::half>();

        int scale_block_id = (i / block_b_y_per_scale * num_block_x + j);
        auto scale_vec = scale.reg.xetla_select<scale_t::block_size_x, 1>(
            scale_block_id * scale_t::block_size_x);
        xetla_vector<dtype_scale, scale_t::block_size_x * vnni_rows>
            scale_vec_vnni;
        for (uint32_t k = 0; k < vnni_rows; ++k) {
          scale_vec_vnni.template select<scale_t::block_size_x, vnni_rows>(k) =
              scale_vec;
        }
        auto scale_lo = SUB * scale_vec_vnni;
        auto scale_hi_0 = scale_vec_vnni / DIV;
        auto scale_hi_1 = scale_vec_vnni * ADD;

        for (uint32_t k = 0; k < packed_block_size_y * 2; ++k) {
          auto sub_lo = f16_lo_matB.template select<block_size_x_b * 2, 1>(
              k * block_size_x_b * 2);
          auto sub_hi = f16_hi_matB.template select<block_size_x_b * 2, 1>(
              k * block_size_x_b * 2);
          // sub_lo = (sub_lo - SUB) * scale_vec_vnni;
          sub_lo = sub_lo * scale_vec_vnni - scale_lo;
          // sub_hi = (sub_hi / DIV - ADD) * scale_vec_vnni;
          sub_hi = sub_hi * scale_hi_0 - scale_hi_1;
        }

        auto dst_blk = matB_acc.reg.xetla_select<matB_acc_t::block_elems, 1>(
            block_id * matB_acc_t::block_elems);

#pragma unroll
        for (uint32_t k = 0; k < packed_block_size_y; ++k) {
          uint32_t dst_offset = k * block_size_x_b * pack_ratio;
          uint32_t src_offset = k * block_size_x_b * 4;
          auto lo_dst =
              dst_blk.template select<4 * block_size_x_b, 1>(dst_offset);
          auto hi_dst = dst_blk.template select<4 * block_size_x_b, 1>(
              dst_offset + 4 * block_size_x_b);
          lo_dst =
              f16_lo_matB.template select<4 * block_size_x_b, 1>(src_offset);
          hi_dst =
              f16_hi_matB.template select<4 * block_size_x_b, 1>(src_offset);
        }
      }
    }
  }
  /// @brief Updates tile base descriptor based on the tid.
  __XETLA_API static void update_sg_tile_tdesc(
      arguments_t& args,
      int32_t sg_idx,
      int32_t sg_idy) {
    int32_t tile_offset_n = sg_idx * sg_tile_n;
    int32_t tile_offset_m = sg_idy * sg_tile_m;

    args.matA_base_desc.update_coord_y(tile_offset_m);
    args.matB_base_desc.update_coord_x(tile_offset_n);
    args.scale_base_desc.update_coord_x(tile_offset_n);
  }
};

/// @} xetla_gemm

} // namespace gpu::xetla::group
