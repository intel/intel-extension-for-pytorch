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

/*
  Indexed Flash Multi-Head Attention Forward
*/

#pragma once

#include <utils/DPCPP.h>
#include <limits>
#include "ifmha_policy.h"
#include "ifmha_utils.h"

namespace gpu::xetla {

namespace fmha {

template <
    typename ifmha_policy,
    typename scalar_t,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsTraining>
class ifmha_forward_t {
 public:
  static_assert((!kIsTraining), "training is not supported yet");

  using accum_t = float;
  using index_t = int32_t;
  static constexpr accum_t kNegInfinity = INFINITY * -1;

  struct arguments_t {
    // Input and output tensors
    scalar_t* Q_ptr; // [1,B,Bm,N,H] - query
    scalar_t* K0_ptr; // [T0,B,1,N,H] - key0
    scalar_t* K1_ptr; // [T1,B,Bm,N,H] - key1
    scalar_t* V0_ptr; // [T0,B,1,N,H] - value0
    scalar_t* V1_ptr; // [T1,B,Bm,N,H] - value1
    index_t* I_ptr; // [T1,B,Bm] - index
    scalar_t* A_ptr = nullptr; // [B,Bm,N,1,T] - alibi
    scalar_t* B_ptr = nullptr; // [B,Bm,N,F,PT] - bias
    uint8_t* Dp_ptr = nullptr; // [B,Bm,N,F,T] - dropout mask
    scalar_t* O_ptr; // [B,Bm,1,N,H] - output

    accum_t dp_prob; // Dropout prob
    accum_t dp_scale; // Dropout scale is computed from dropout prob
    accum_t sm_scale; // Softmax scale

    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uH;
    uint32_t uT0;
    uint32_t uT1;
    uint32_t uAT;
    uint32_t uPT;

    inline arguments_t(
        scalar_t* query,
        scalar_t* key0,
        scalar_t* key1,
        scalar_t* value0,
        scalar_t* value1,
        index_t* index,
        scalar_t* alibi,
        scalar_t* bias,
        uint8_t* dropout,
        accum_t dropout_prob,
        accum_t sm_scale,
        scalar_t* out,
        uint32_t num_batches,
        uint32_t beam,
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t kv_len0,
        uint32_t kv_len1,
        uint32_t padded_alibi,
        uint32_t padded_kvlen)
        : Q_ptr(query),
          K0_ptr(key0),
          K1_ptr(key1),
          V0_ptr(value0),
          V1_ptr(value1),
          I_ptr(index),
          A_ptr(alibi),
          B_ptr(bias),
          Dp_ptr(dropout),
          dp_prob(dropout_prob),
          dp_scale(1.f / (1.f - dropout_prob)),
          sm_scale(sm_scale),
          O_ptr(out),
          uB(num_batches),
          uN(num_heads),
          uH(head_size),
          uT0(kv_len0),
          uT1(kv_len1),
          uAT(padded_alibi),
          uPT(padded_kvlen) {}
  };

 private:
  // -------------------- // Compute policy // -------------------- //
  static constexpr uint32_t accum_step_bmbc = ifmha_policy::accum_step_bmbc;
  static constexpr uint32_t stages_bmbc = ifmha_policy::stages_bmbc;
  static constexpr uint32_t sync_freq_bmbc = ifmha_policy::sync_freq_bmbc;

  static constexpr uint32_t accum_step_bmhm = ifmha_policy::accum_step_bmhm;
  static constexpr uint32_t stages_bmhm = ifmha_policy::stages_bmhm;
  static constexpr uint32_t sync_freq_bmhm = ifmha_policy::sync_freq_bmhm;

  using compute_attr = group::compute_attr_t<scalar_t, scalar_t, accum_t>;

  // ---------------- // Tile shape and Threads // ---------------- //
  static constexpr uint32_t Beams = ifmha_policy::Beams;
  static constexpr uint32_t kBm = ifmha_policy::kBm;

  static constexpr uint32_t kBc = ifmha_policy::kBc;
  static constexpr uint32_t kHm = ifmha_policy::kHm;
  static constexpr uint32_t kSgBc = ifmha_policy::kSgBc;
  static constexpr uint32_t kSgHm = ifmha_policy::kSgHm;

  using tile_shape_BmBc = group::tile_shape_t<kBc, kBm, kSgBc, kBm>;
  using tile_shape_BmHm = group::tile_shape_t<kHm, kBm, kSgHm, kBm>;

  using work_group_t = typename tile_shape_BmBc::work_group_t;
  static constexpr uint32_t wg_size_x = tile_shape_BmBc::wg_size_x;

  static_assert(
      kHm / kSgHm == kBc / kSgBc,
      "wg_size_x must be the same between Hm and Bc");
  static_assert(
      Beams * wg_size_x <= 32,
      "The number of threads should be less than 32!");

  // -------------------- // 1D load config // -------------------- //
  static constexpr uint32_t max_load_bytes = ifmha_policy::max_load_bytes;
  static constexpr uint32_t reduce_size = ifmha_policy::reduce_size;

  static constexpr uint32_t max_load_size = max_load_bytes / sizeof(scalar_t);
  static constexpr uint32_t accum_step_1d =
      kHm < max_load_size ? kHm : max_load_size;

  static_assert(
      accum_step_1d % reduce_size == 0,
      "accum_step_1d should be a multiple of reduce_size");

  // --------------------- // Memory desc // ---------------------- //
  using mem_desc_Qi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_QiL_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Kj_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_Vj_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Oi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Ij_t =
      mem_desc_t<index_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Bij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Pij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;

  using imem_desc_Kj_t = imem_desc_t<scalar_t, kSgBc, 1, accum_step_1d>;
  using imem_desc_Vj_t = imem_desc_t<scalar_t, kSgBc, 1, accum_step_1d>;
  using matQ_tile_desc_t = subgroup::tile_desc_t<kHm, 1, kHm, 1>;
  using matQ_t = subgroup::tile_t<scalar_t, matQ_tile_desc_t>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t slm_size_Qi = 0;

  static constexpr uint32_t slm_size_softmax =
      (wg_size_x > 1) ? Beams * wg_size_x * sizeof(accum_t) : 0;
  static constexpr uint32_t slm_size_Pij = Beams * kBc * sizeof(scalar_t);
  static constexpr uint32_t slm_size_PV = Beams * kHm * sizeof(accum_t);

  // Slm addr to store intermediate results
  static constexpr uint32_t Qi_slm = 0;
  static constexpr uint32_t softmax_slm = Qi_slm + slm_size_Qi;
  static constexpr uint32_t Pij_slm = softmax_slm + slm_size_softmax;
  static constexpr uint32_t PV_slm = Pij_slm;

  static constexpr uint32_t nbarrier_cnt_x = (wg_size_x > 1) ? Beams : 0;
  static constexpr uint32_t nbarrier_cnt_ring = 0;
  static constexpr uint32_t nbarrier_cnt =
      std::max(nbarrier_cnt_x, nbarrier_cnt_ring);

  // ======================== // Context // ======================= //

  /// @brief Used to store variables in the ifmha loops
  struct context_t {
    uint32_t batch_id;
    uint32_t beam_id;
    uint32_t head_id;
    uint32_t sg_idx;
    work_group_t g;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier;
    // softmax statistics
    xetla_vector<accum_t, kBm> softmax_m;
    xetla_vector<accum_t, kBm> softmax_l;
    // mem desc variables
    mem_desc_Qi_t desc_Qi;
    mem_desc_QiL_t desc_QiL;
    mem_desc_Kj_t desc_Kj;
    mem_desc_Vj_t desc_Vj;
    mem_desc_Oi_t desc_Oi;
    mem_desc_Ij_t desc_Ij;
    mem_desc_Bij_t desc_Bij;
    mem_desc_Pij_t desc_Pij;

    imem_desc_Kj_t idesc_Kj;
    imem_desc_Vj_t idesc_Vj;
    uint32_t bias_base_offset;
    uint32_t alibi_base_offset;

    inline context_t() = default;

    /// @brief Initialize invariant variables in the ifmha loop
    inline void init_context(xetla_exec_item<2>& ei, arguments_t& args) {
      batch_id = ei.get_group(0);
      head_id = ei.get_group(1);
      beam_id = ei.get_local_id(0);
      sg_idx = ei.get_local_id(1);
      g.init(sg_idx);

      nbarrier.init_nbarrier(beam_id, nbarrier_role::producer_consumer);

      softmax_m = kNegInfinity;
      softmax_l = 0.f;

      // mem desc variables
      int32_t start_x = head_id * args.uH;
      int32_t start_y = batch_id * Beams + beam_id;
      uint32_t width = start_x + args.uH;
      uint32_t height = start_y + 1;
      uint32_t pitch = args.uN * args.uH;
      desc_Qi.init(args.Q_ptr, {width, height, pitch}, {start_x, start_y});
      desc_Oi.init(args.O_ptr, {width, height, pitch}, {start_x, start_y});

      start_y = beam_id;
      desc_QiL.init(Qi_slm, {kHm, Beams, kHm}, {0, start_y});
      desc_Pij.init(Pij_slm, {kBc, Beams, kBc}, {0, start_y});

      idesc_Kj.init(args.K1_ptr, args.uH, args.uH);
      idesc_Vj.init(args.V1_ptr, args.uH, args.uH);

      // (b, bm, N, 1, t)
      if constexpr (kUseAlibi) {
        alibi_base_offset = batch_id * Beams * args.uN * args.uAT +
            beam_id * args.uN * args.uAT + head_id * args.uAT;
      }
      // (b, bm, N, 1, t)
      if constexpr (kUseBias) {
        bias_base_offset = batch_id * Beams * args.uN * args.uPT +
            beam_id * args.uN * args.uPT + head_id * args.uPT;
      };
    }

    /// @brief Set context for buffer 0
    inline void set_context0(arguments_t& args, int32_t start_T) {
      int32_t start_x = head_id * args.uH;
      int32_t start_y = batch_id * args.uT0 + start_T;
      uint32_t width = start_x + args.uH;
      uint32_t height = (start_y + kBc) < (batch_id + 1) * args.uT0
          ? (start_y + kBc)
          : (batch_id + 1) * args.uT0;
      uint32_t pitch = args.uN * args.uH;

      desc_Kj.init(args.K0_ptr, {height, width, pitch}, {start_y, start_x});
      desc_Vj.init(args.V0_ptr, {width, height, pitch}, {start_x, start_y});
    }

    /// @brief Set context for buffer 1
    inline void set_context1(arguments_t& args, int32_t start_T) {
      // TODO: fix this hardcode
      using index_tile_desc_t = subgroup::tile_desc_t<1, kSgBc, 1, 16>;
      using index_tile_t = subgroup::tile_t<index_t, index_tile_desc_t>;
      using index_payload_t = subgroup::mem_payload_t<
          index_t,
          index_tile_desc_t,
          msg_type::block_2d,
          mem_desc_Ij_t::layout,
          mem_desc_Ij_t::space>;
      index_tile_t index_tile;
      index_payload_t index_payload;

      int32_t start_x = batch_id * Beams + beam_id;
      int32_t start_y = (start_T + sg_idx * kSgBc) < args.uT1
          ? (start_T + sg_idx * kSgBc)
          : args.uT1;
      uint32_t width = start_x + 1;
      uint32_t height =
          (start_y + kSgBc) < args.uT1 ? (start_y + kSgBc) : args.uT1;
      uint32_t pitch = args.uB * Beams;
      desc_Ij.init(args.I_ptr, {width, height, pitch}, {start_x, start_y});

      index_payload.init(desc_Ij);
      subgroup::tile_load(index_tile, index_payload);

      xetla_vector<int32_t, kSgBc> index =
          xetla_vector_gen<int32_t, kSgBc>(0, 1);

      index = index + start_y;
      index = index * args.uB * Beams * args.uN + batch_id * Beams * args.uN +
          index_tile.reg * args.uN + head_id;

      uint32_t total = height - start_y;
      idesc_Kj.init_index(index, total);
      idesc_Vj.init_index(index, total);
    }
  };

  context_t ctx;

  // ======================= // gemm_Sij // ======================= //

  using perf_tuning_knob_bmbc =
      group::perf_tuning_knob_t<accum_step_bmbc, stages_bmbc, sync_freq_bmbc>;
  using compute_policy_bmbc =
      group::compute_policy_default_xmx<compute_attr, perf_tuning_knob_bmbc>;
  using brgemm_Sij_t = group::brgemm_t<
      compute_policy_bmbc,
      tile_shape_BmBc,
      mem_desc_QiL_t,
      mem_desc_Kj_t>;
  using matSij_t = typename brgemm_Sij_t::matAcc_t;

  template <
      typename matA_t,
      typename matB_t,
      typename matSrc_t,
      typename matDst_t>
  inline void xmx_mma(matA_t& a, matB_t& b, matSrc_t& src, matDst_t& dst) {
    using dtype_a = typename matA_t::dtype;
    using dtype_b = typename matB_t::dtype;
    using dtype_src = typename matSrc_t::dtype;
    using dtype_dst = typename matDst_t::dtype;
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

    static constexpr int32_t num_block_n = matDst_t::num_block_x;
    static constexpr int32_t num_block_m = matDst_t::num_block_y;
    static constexpr int32_t num_block_k = tile_size_k / block_size_k;
    static constexpr int32_t num_block = num_block_m * num_block_n;

    static constexpr int32_t mma_m = tile_shape_BmBc::sg_tile_size_y;
    static constexpr int32_t mma_k = 8;
    static_assert(
        tile_size_m % mma_m == 0, "tile_size_m shoud be a multiple of mma_m");
    static_assert(mma_m <= 8);

    constexpr int32_t a_mma_elems = mma_m * a_block_size_x;
    constexpr int32_t c_mma_elems = mma_m * block_size_n;
#pragma unroll
    for (int j = 0; j < num_block_n; j++) {
#pragma unroll
      for (int i = 0; i < tile_size_m / block_size_m; i++) {
        auto src_block = src.reg.xetla_select<block_elems, 1>(
            (i * num_block_n + j) * block_elems);
        auto dst_block = dst.reg.xetla_select<block_elems, 1>(
            (i * num_block_n + j) * block_elems);
#pragma unroll
        for (int mma_i = 0; mma_i < block_size_m / mma_m; mma_i++) {
          auto src_sub_blk =
              src_block.xetla_select<c_mma_elems, 1>(mma_i * c_mma_elems);
          auto dst_sub_blk =
              dst_block.xetla_select<c_mma_elems, 1>(mma_i * c_mma_elems);
          { // k=0
            auto a_block = a.reg.xetla_select<a_block_elems, 1>(
                (i * num_block_k) * a_block_elems);
            auto a_sub_blk =
                a_block.xetla_select<a_mma_elems, 1>(mma_i * a_mma_elems);
            auto b_sub_blk =
                b.reg.xetla_select<b_block_elems, 1>(j * b_block_elems);
            dst_sub_blk = xetla_mma<
                gpu::xetla::detail::mma_argument_type<dtype_b>(),
                gpu::xetla::detail::mma_argument_type<dtype_a>(),
                mma_k,
                mma_m,
                dtype_src,
                uint32_t,
                uint32_t,
                c_mma_elems,
                b_block_elems / (sizeof(uint32_t) / sizeof(dtype_b)),
                a_mma_elems / (sizeof(uint32_t) / sizeof(dtype_a))>(
                src_sub_blk,
                b_sub_blk.xetla_format<uint32_t>(),
                a_sub_blk.xetla_format<uint32_t>());
          }
#pragma unroll
          for (int k = 1; k < num_block_k; k++) {
            auto a_block = a.reg.xetla_select<a_block_elems, 1>(
                (i * num_block_k + k) * a_block_elems);
            auto a_sub_blk =
                a_block.xetla_select<a_mma_elems, 1>(mma_i * a_mma_elems);
            auto b_sub_blk = b.reg.xetla_select<b_block_elems, 1>(
                (j + k * num_block_n) * b_block_elems);
            dst_sub_blk = xetla_mma<
                gpu::xetla::detail::mma_argument_type<dtype_b>(),
                gpu::xetla::detail::mma_argument_type<dtype_a>(),
                mma_k,
                mma_m,
                dtype_src,
                uint32_t,
                uint32_t,
                c_mma_elems,
                b_block_elems / (sizeof(uint32_t) / sizeof(dtype_b)),
                a_mma_elems / (sizeof(uint32_t) / sizeof(dtype_a))>(
                dst_sub_blk,
                b_sub_blk.xetla_format<uint32_t>(),
                a_sub_blk.xetla_format<uint32_t>());
          }
        }
      }
    }
    if constexpr (num_block_k > 1) {
      constexpr uint32_t last_uint16_idx =
          tile_elems * sizeof(dtype_dst) / sizeof(uint16_t) - 1;
      xetla_wait(dst.reg.xetla_format<uint16_t>()[last_uint16_idx]);
    }
  }

  inline void gemm0_Sij(matQ_t& matQ, matSij_t& matSij, arguments_t& args) {
    constexpr uint32_t tile_size_x_a = accum_step_bmbc;
    constexpr uint32_t tile_size_y_a = tile_shape_BmBc::sg_tile_size_y;
    constexpr uint32_t block_size_x_a = 32 / sizeof(scalar_t);
    constexpr uint32_t block_size_y_a = tile_size_y_a;
    constexpr uint32_t tile_size_x_b = tile_shape_BmBc::sg_tile_size_x;
    constexpr uint32_t tile_size_y_b = accum_step_bmbc;
    constexpr uint32_t block_size_x_b = 16;
    constexpr uint32_t block_size_y_b = 32 / sizeof(scalar_t);

    constexpr mem_layout layout_b = mem_layout::col_major;
    constexpr tdesc_update_dir update_dir_b =
        (layout_b == mem_layout::col_major) ? tdesc_update_dir::x_dir
                                            : tdesc_update_dir::y_dir;

    uint32_t loop_count = (args.uH + accum_step_bmbc - 1) / accum_step_bmbc;

    using matA_tile_desc_t = subgroup::tile_desc_t<
        tile_size_x_a,
        tile_size_y_a,
        block_size_x_a,
        block_size_y_a,
        reg_layout::tiled>;
    using matA_t = subgroup::tile_t<scalar_t, matA_tile_desc_t>;
    using matB_tile_desc_t = subgroup::tile_desc_t<
        tile_size_x_b,
        tile_size_y_b,
        block_size_x_b,
        block_size_y_b,
        reg_layout::vnni_tiled>;
    using matB_t = subgroup::tile_t<scalar_t, matB_tile_desc_t>;
    using matB_payload_t = subgroup::mem_payload_t<
        scalar_t,
        matB_tile_desc_t,
        subgroup::msg_type_v<matB_tile_desc_t, mem_space::global>,
        layout_b,
        mem_space::global,
        gpu_arch::Xe>;
    using matB_prefetch_payload_t = subgroup::ext::prefetch_payload_t<
        scalar_t,
        subgroup::tile_desc_t<kSgBc, accum_step_bmbc, 1, 1>,
        layout_b,
        mem_space::global,
        tile_shape_BmBc::wg_size_y,
        gpu_arch::Xe>;

    mem_desc_Kj_t matB_mem_desc = ctx.desc_Kj;
    int32_t sg_idx = ctx.g.get_id() % tile_shape_BmBc::wg_size_x;
    int32_t sg_idy = ctx.g.get_id() / tile_shape_BmBc::wg_size_x;
    int32_t sg_tile_offset_x = sg_idx * tile_shape_BmBc::sg_tile_size_x;
    matB_mem_desc.update_coord_x(sg_tile_offset_x);

    matB_payload_t matB_payload(matB_mem_desc);
    matB_prefetch_payload_t matB_prefetch_payload(matB_mem_desc, sg_idy);
    xetla_nbarrier_t<tile_shape_BmBc::wg_size_y, tile_shape_BmBc::wg_size_y>
        nbarrier_b;
    nbarrier_b.init_nbarrier(
        sg_idx + nbarrier_cnt, nbarrier_role::producer_consumer);

    matA_t matA;
    matB_t matB;
#pragma unroll
    for (int i = 0; i < stages_bmbc; i++) {
      subgroup::ext::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          matB_prefetch_payload);
      matB_prefetch_payload.template update_tdesc<update_dir_b>(
          matB_t::tile_size_y);
    }

    for (int i = 0; i < loop_count; i++) {
      if constexpr (sync_freq_bmbc > 0) {
        if constexpr ((i % sync_freq_bmbc) == 0) {
          if constexpr (tile_shape_BmBc::wg_size_y > 1) {
            nbarrier_b.arrive();
          }
        }
      }
      matA.reg =
          matQ.reg.xetla_select<matA_t::tile_elems, 1>(matA_t::tile_elems * i);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matB, matB_payload);
      if constexpr (stages_bmbc != 0) {
        subgroup::ext::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            matB_prefetch_payload);
      }
      matB_payload.template update_tdesc<update_dir_b>(matB_t::tile_size_y);
      if constexpr (stages_bmbc != 0) {
        matB_prefetch_payload.template update_tdesc<update_dir_b>(
            matB_t::tile_size_y);
      }
      SW_BARRIER();
      xmx_mma(matA, matB, matSij, matSij);
      SW_BARRIER();
      if constexpr (sync_freq_bmbc > 0) {
        if constexpr ((i % sync_freq_bmbc) == 0) {
          if constexpr (tile_shape_BmBc::wg_size_y > 1) {
            nbarrier_b.wait();
          }
        }
      }
    }
  }

  /// @brief gemm1_Sij is used to compute Qi x Kj1
  inline void gemm1_Sij(matQ_t& matQ, matSij_t& matSij, arguments_t& args) {
    constexpr uint32_t block_size_x = matSij_t::block_size_x;
    constexpr uint32_t num_block_x = matSij_t::num_block_x;
    constexpr uint32_t block_elems = matSij_t::block_elems;

    using matQi_tile_desc_t = subgroup::tile_desc_t<accum_step_1d, 1, 16, 1>;
    using matQi_payload_t = subgroup::mem_payload_t<
        scalar_t,
        matQi_tile_desc_t,
        msg_type::block_1d,
        mem_desc_QiL_t::layout,
        mem_desc_QiL_t::space>;
    using matQi_t = subgroup::tile_t<scalar_t, matQi_tile_desc_t>;
    using matQi_acc_t = subgroup::tile_t<accum_t, matQi_tile_desc_t>;

    using matKj_tile_desc_t = subgroup::tile_desc_t<accum_step_1d, 1, 16, 1>;
    using matKj_acc_t = subgroup::tile_t<accum_t, matKj_tile_desc_t>;

    using matQK_tile_desc_t = subgroup::
        tile_desc_t<reduce_size, block_size_x, reduce_size, block_size_x>;
    using matQK_acc_t = subgroup::tile_t<accum_t, matQK_tile_desc_t>;

    matQi_t matQi;
    matQi_acc_t matQi_acc;
    matKj_acc_t matKj_acc;

#pragma unroll
    for (int i = 0; i < stages_bmbc; i++) {
      ctx.idesc_Kj.iprefetch_tile();
      ctx.idesc_Kj.update_prefetch_tdesc();
    }
    uint32_t loop_count = (args.uH + accum_step_1d - 1) / accum_step_1d;

    for (int istep = 0; istep < loop_count; istep++) {
#pragma unroll
      for (int i = 0; i < 1; i++) {
        matQi.reg =
            matQ.reg.xetla_select<accum_step_1d, 1>(istep * accum_step_1d);
        subgroup::elemwise_cvt(matQi_acc, matQi);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
          matQK_acc_t matQK_acc;
#pragma unroll
          for (int k = 0; k < block_size_x; k++) {
            // load Kj from buffer 1
            ctx.idesc_Kj.iload_tile(matKj_acc);
            ctx.idesc_Kj.iprefetch_tile();
            SW_BARRIER();
            ctx.idesc_Kj.update_tdesc();
            ctx.idesc_Kj.update_prefetch_tdesc();
            SW_BARRIER();
            // compute
            auto QK_sub =
                matQK_acc.reg.xetla_select<reduce_size, 1>(k * reduce_size);
            xetla_vector<accum_t, accum_step_1d> tmp =
                matQi_acc.reg * matKj_acc.reg;
            QK_sub = partial_reduce<accum_t, accum_step_1d, reduce_size>(tmp);
          }

          auto src_sub = matSij.reg.xetla_select<block_size_x, 1>(
              j * block_elems + i * block_size_x);
          src_sub += recur_col_reduce<
              reduce_op::sum,
              accum_t,
              reduce_size,
              block_size_x>(matQK_acc.reg);
        }
      }
    }
  }

  // ======================= // gemm_Oi // ======================= //
  using perf_tuning_knob_bmhm =
      group::perf_tuning_knob_t<accum_step_bmhm, stages_bmhm, sync_freq_bmhm>;
  using compute_policy_bmhm =
      group::compute_policy_default_xmx<compute_attr, perf_tuning_knob_bmhm>;
  using brgemm_Oi_t = group::brgemm_t<
      compute_policy_bmhm,
      tile_shape_BmHm,
      mem_desc_Pij_t,
      mem_desc_Vj_t>;
  using matOi_t = typename brgemm_Oi_t::matAcc_t;

  // ====================== // prefetch_V0j // ===================== //

  inline void prefetch_V0() {
    using matVj_prefetch_payload_t = subgroup::ext::prefetch_payload_t<
        scalar_t,
        subgroup::tile_desc_t<kSgHm, stages_bmhm * accum_step_bmhm, 1, 1>,
        mem_layout::row_major,
        mem_space::global,
        1>;

    mem_desc_Vj_t desc_pre_Vj(ctx.desc_Vj);
    desc_pre_Vj.update_coord_x(ctx.sg_idx * kSgHm);
    matVj_prefetch_payload_t matVj_prefetch_payload(desc_pre_Vj, 0);

    subgroup::ext::tile_prefetch<cache_hint::cached, cache_hint::cached>(
        matVj_prefetch_payload);
  }

  /// @brief gemm_Oi is used to compute Oi += Pij x Vj
  /// # [Bm,Bc] x [Bc,H] = [Bm,Hm]
#if 0
  inline void gemm0_Oi(matOi_t &matOi, arguments_t &args, int32_t start_T) {
    using brgemm_args_t = typename brgemm_Oi_t::arguments_t;

    uint32_t remain_T = args.uT0 - start_T;
    uint32_t boundary_k = remain_T > kBc ? kBc : remain_T;
    uint32_t loop_count = (boundary_k + accum_step_bmhm - 1) / accum_step_bmhm;

    // Gemm to comput Oi
    brgemm_Oi_t brgemm;
    brgemm_args_t brgemm_args(ctx.desc_Pij, ctx.desc_Vj, loop_count);
    brgemm(ctx.g, matOi, brgemm_args, 0, /* nbarrier_base */ nbarrier_cnt);
  }
#else
  inline void gemm0_Oi(matOi_t& matOi, arguments_t& args, int32_t start_T) {
    using matPi_tile_desc_t = subgroup::tile_desc_t<kBc, 1, kBc, 1>;
    using matPi_t = subgroup::tile_t<scalar_t, matPi_tile_desc_t>;
    using matPi_load_t = subgroup::mem_payload_t<
        scalar_t,
        matPi_t,
        msg_type::block_1d,
        mem_desc_Pij_t::layout,
        mem_desc_Pij_t::space>;

    mem_desc_Pij_t desc_Pi_load(ctx.desc_Pij);
    matPi_load_t matPi_load(desc_Pi_load);
    matPi_t matP;
    subgroup::tile_load(matP, matPi_load);

    constexpr uint32_t tile_size_x_a = accum_step_bmhm;
    constexpr uint32_t tile_size_y_a = tile_shape_BmHm::sg_tile_size_y;
    constexpr uint32_t block_size_x_a = 32 / sizeof(scalar_t);
    constexpr uint32_t block_size_y_a = tile_size_y_a;
    constexpr uint32_t tile_size_x_b = tile_shape_BmHm::sg_tile_size_x;
    constexpr uint32_t tile_size_y_b = accum_step_bmhm;
    constexpr uint32_t block_size_x_b = 16;
    constexpr uint32_t block_size_y_b = 32 / sizeof(scalar_t);

    constexpr mem_layout layout_b = mem_layout::row_major;
    constexpr tdesc_update_dir update_dir_b =
        (layout_b == mem_layout::col_major) ? tdesc_update_dir::x_dir
                                            : tdesc_update_dir::y_dir;
    constexpr uint32_t loop_count =
        (kBc + accum_step_bmhm - 1) / accum_step_bmhm;

    using matA_tile_desc_t = subgroup::tile_desc_t<
        tile_size_x_a,
        tile_size_y_a,
        block_size_x_a,
        block_size_y_a,
        reg_layout::tiled>;
    using matA_t = subgroup::tile_t<scalar_t, matA_tile_desc_t>;
    using matB_tile_desc_t = subgroup::tile_desc_t<
        tile_size_x_b,
        tile_size_y_b,
        block_size_x_b,
        block_size_y_b,
        reg_layout::vnni_tiled>;
    using matB_t = subgroup::tile_t<scalar_t, matB_tile_desc_t>;
    using matB_payload_t = subgroup::mem_payload_t<
        scalar_t,
        matB_tile_desc_t,
        msg_type::block_2d,
        layout_b,
        mem_space::global,
        gpu_arch::Xe>;
    using matB_prefetch_payload_t = subgroup::ext::prefetch_payload_t<
        scalar_t,
        subgroup::tile_desc_t<kSgBc, accum_step_bmhm, 1, 1>,
        layout_b,
        mem_space::global,
        tile_shape_BmHm::wg_size_y,
        gpu_arch::Xe>;

    mem_desc_Vj_t matB_mem_desc(ctx.desc_Vj);
    int32_t sg_idx = ctx.g.get_id() % tile_shape_BmHm::wg_size_x;
    int32_t sg_idy = ctx.g.get_id() / tile_shape_BmHm::wg_size_x;
    int32_t sg_tile_offset_x = sg_idx * tile_shape_BmHm::sg_tile_size_x;
    matB_mem_desc.update_coord_x(sg_tile_offset_x);

    matB_payload_t matB_payload(matB_mem_desc);
    matB_prefetch_payload_t matB_prefetch_payload(matB_mem_desc, sg_idy);
    xetla_nbarrier_t<tile_shape_BmHm::wg_size_y, tile_shape_BmHm::wg_size_y>
        nbarrier_b;
    nbarrier_b.init_nbarrier(
        sg_idx + nbarrier_cnt, nbarrier_role::producer_consumer);

    matA_t matA;
    matB_t matB;
#pragma unroll
    for (int i = 0; i < stages_bmbc; i++) {
      subgroup::ext::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          matB_prefetch_payload);
      matB_prefetch_payload.template update_tdesc<update_dir_b>(
          matB_t::tile_size_y);
    }
#pragma unroll
    for (int i = 0; i < loop_count; i++) {
      if constexpr (sync_freq_bmbc > 0) {
        if constexpr ((i % sync_freq_bmbc) == 0) {
          if constexpr (tile_shape_BmHm::wg_size_y > 1) {
            nbarrier_b.arrive();
          }
        }
      }
      matA.reg =
          matP.reg.xetla_select<matA_t::tile_elems, 1>(matA_t::tile_elems * i);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matB, matB_payload);
      if constexpr (stages_bmbc != 0) {
        subgroup::ext::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            matB_prefetch_payload);
      }
      matB_payload.template update_tdesc<update_dir_b>(matB_t::tile_size_y);
      if constexpr (stages_bmbc != 0) {
        matB_prefetch_payload.template update_tdesc<update_dir_b>(
            matB_t::tile_size_y);
      }
      SW_BARRIER();
      xmx_mma(matA, matB, matOi, matOi);
      SW_BARRIER();
      if constexpr (sync_freq_bmbc > 0) {
        if constexpr ((i % sync_freq_bmbc) == 0) {
          if constexpr (tile_shape_BmHm::wg_size_y > 1) {
            nbarrier_b.wait();
          }
        }
      }
    }
  }
#endif

  using matPij_t = matSij_t;

  inline void prefetch_v1() {
#pragma unroll
    for (int i = 0; i < stages_bmhm; i++) {
      ctx.idesc_Vj.iprefetch_tile();
      ctx.idesc_Vj.update_prefetch_tdesc();
    }
  }

  /// @brief gemm_Oi is used to compute Oi += Pij x Vj
  inline void gemm1_Oi(arguments_t& args, matPij_t& matPij, matOi_t& matOi) {
    constexpr uint32_t block_size_x = matPij_t::block_size_x;
    constexpr uint32_t num_block_x = matPij_t::num_block_x;
    constexpr uint32_t block_elems = matPij_t::block_elems;

    using matVj_tile_desc_t = subgroup::tile_desc_t<accum_step_1d, 1, 16, 1>;
    using matVj_acc_t = subgroup::tile_t<accum_t, matVj_tile_desc_t>;

    using matPV_tile_desc_t = subgroup::tile_desc_t<kHm, 1, kHm, 1>;
    using matPV_acc_t = subgroup::tile_t<accum_t, matPV_tile_desc_t>;

    matPV_acc_t matPV_acc(0);

    uint32_t loop_count = (args.uH + accum_step_1d - 1) / accum_step_1d;

    for (int istep = 0; istep < loop_count; istep++) {
      matVj_acc_t matVj_acc;
#pragma unroll
      for (int i = 0; i < 1; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
#pragma unroll
          for (int k = 0; k < block_size_x; k++) {
            // load Vj from buffer 1
            ctx.idesc_Vj.iload_tile(matVj_acc);
            ctx.idesc_Vj.iprefetch_tile();
            SW_BARRIER();
            ctx.idesc_Vj.update_tdesc();
            ctx.idesc_Vj.update_prefetch_tdesc();
            SW_BARRIER();

            accum_t tmp = matPij.reg.xetla_select<1, 1>(
                j * block_elems + i * block_size_x + k);

            auto PV_sub = matPV_acc.reg.xetla_select<accum_step_1d, 1>(
                i * kHm + istep * accum_step_1d);
            PV_sub += tmp * matVj_acc.reg;
          }
        }
      }
    }

    if constexpr (wg_size_x > 1) {
      using matX_tile_desc_t = subgroup::tile_desc_t<kSgHm, 1, 16, 1>;
      using matX_payload_t = subgroup::mem_payload_t<
          accum_t,
          matX_tile_desc_t,
          msg_type::block_1d,
          mem_layout::row_major,
          mem_space::local>;
      using matX_acc_t = subgroup::tile_t<accum_t, matX_tile_desc_t>;

      // xetla_nbarrier_t<1, 1> nbarrier_producer;
      // xetla_nbarrier_t<1, 1> nbarrier_consumer;
      // nbarrier_producer.init_nbarrier(ctx.sg_idx, nbarrier_role::producer);
      // nbarrier_consumer.init_nbarrier((ctx.sg_idx + 1) % wg_size_x,
      // nbarrier_role::consumer);

      matX_acc_t matX_acc;
      int offset = ctx.sg_idx * kSgHm;

      matX_acc.reg = matPV_acc.reg.xetla_select<kSgHm, 1>(offset);
      matX_payload_t matX_payload(
          PV_slm, kHm, ctx.beam_id + 1, kHm, offset, ctx.beam_id);
      subgroup::tile_store(matX_acc, matX_payload);

      xetla_fence<memory_kind::shared_local>();
      // nbarrier_producer.arrive();
      // nbarrier_consumer.arrive_wait();
      ctx.nbarrier.arrive_wait();
#pragma unroll
      for (int i = 1; i < wg_size_x; i++) {
        int id = (ctx.sg_idx + i) % wg_size_x;
        offset = id * kSgHm;

        for (int j = 0; j < 1; j++) {
          matX_payload_t matX_payload(
              PV_slm, kHm, ctx.beam_id + 1, kHm, offset, ctx.beam_id);
          subgroup::tile_load(matX_acc, matX_payload);

          auto PV_sub = matPV_acc.reg.xetla_select<kSgHm, 1>(j * kHm + offset);
          matX_acc.reg += PV_sub;

          subgroup::tile_store(matX_acc, matX_payload);
        }

        xetla_fence<memory_kind::shared_local>();
        ctx.nbarrier.arrive_wait();

        // nbarrier_producer.arrive();
        // nbarrier_consumer.arrive_wait();
      }

      constexpr uint32_t block_size_x = matOi_t::block_size_x;
      constexpr uint32_t num_block_x = matOi_t::num_block_x;
      constexpr uint32_t block_elems = matOi_t::block_elems;

      offset = ctx.sg_idx * kSgHm;
#pragma unroll
      for (int i = 0; i < 1; i++) {
        matX_payload_t matX_payload(
            PV_slm, kHm, ctx.beam_id + 1, kHm, offset, ctx.beam_id);
        subgroup::tile_load(matX_acc, matX_payload);

        for (int j = 0; j < num_block_x; j++) {
          auto dst_sub = matOi.reg.xetla_select<block_size_x, 1>(
              j * block_elems + i * block_size_x);
          auto src_sub =
              matX_acc.reg.xetla_select<block_size_x, 1>(j * block_size_x);
          dst_sub += src_sub;
        }
      }
    } else {
      constexpr uint32_t block_size_x = matOi_t::block_size_x;
      constexpr uint32_t num_block_x = matOi_t::num_block_x;
      constexpr uint32_t block_elems = matOi_t::block_elems;

      for (int i = 0; i < 1; i++) {
        for (int j = 0; j < num_block_x; j++) {
          auto dst_sub = matOi.reg.xetla_select<block_size_x, 1>(
              j * block_elems + i * block_size_x);
          auto src_sub = matPV_acc.reg.xetla_select<block_size_x, 1>(
              i * kHm + j * block_size_x);
          dst_sub += src_sub;
        }
      }
    }
  }

  // ====================== // pre_softmax // ===================== //

  /// @brief softmax pre_processing function
  inline void pre_softmax(
      matSij_t& matSij,
      matSij_t& matAlibi,
      matSij_t& matBias,
      arguments_t& args,
      int32_t start_T,
      int32_t end_T) {
    // Multiply by softmax scaling factor
    matSij.reg *= args.sm_scale;

    if constexpr (kUseAlibi) {
      matSij.reg += matAlibi.reg;
    }

    if constexpr (kUseBias) {
      matSij.reg += matBias.reg;
    }

    // padding mask to the tail, if needed.
    using tile_mask = tile_mask_t<matSij_t>;

    uint32_t sg_start_T = start_T + ctx.sg_idx * kSgBc;
    uint32_t num_keep = std::max(int(end_T) - int(sg_start_T), 0);
    if (num_keep < kSgBc) {
      tile_mask::padding_mask(matSij, num_keep);
    }
  }

  // ====================== // softmax_fwd // ===================== //

  /// @brief softmax_fwd is used to do softmax.
  inline void softmax_fwd(matSij_t& matSij, matOi_t& matOi) {
    using wg_max_t = group_row_reduce_t<matSij_t, wg_size_x, reduce_op::max>;
    using wg_sum_t = group_row_reduce_t<matSij_t, wg_size_x, reduce_op::sum>;

    uint32_t softmax_slm_start =
        softmax_slm + ctx.beam_id * wg_size_x * sizeof(accum_t);

    // compute new m
    wg_max_t wg_max(ctx.sg_idx, ctx.beam_id, softmax_slm_start);
    xetla_vector<accum_t, kBm> m_new = wg_max(matSij);
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive();
    m_new = xetla_max<accum_t, kBm>(m_new, ctx.softmax_m);

    // correct old l
    ctx.softmax_l *= xetla_exp<accum_t, kBm>(ctx.softmax_m - m_new);
    // compute Pij
    auto delta = matSij;
    subgroup::tile_broadcast_op<subgroup::tile_minus, matSij_t>(delta, m_new);
    // matSij.reg = xetla_exp<accum_t, kSgBc>(matSij.reg - m_new);

    matSij_t mat_zeros(0);
    constexpr int elems = matSij_t::tile_desc::tile_elems;
    // xetla_mask<elems> mask = matAccSij->reg < (INFINITY * -1) ||
    // matAccSij->reg > INFINITY;
    xetla_mask<elems> mask = delta.reg < -65400.f;
    (matSij.reg)
        .xetla_merge(mat_zeros.reg, xetla_exp<accum_t>(delta.reg), mask);
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.wait();

    // compute new l
    wg_sum_t wg_sum(ctx.sg_idx, ctx.beam_id, softmax_slm_start);
    xetla_vector<accum_t, kBm> l_new = wg_sum(matSij);
    l_new += ctx.softmax_l;

    // rescale Pij and Oi
    subgroup::tile_broadcast_op<subgroup::tile_div, matSij_t>(matSij, l_new);
    xetla_vector<accum_t, kBm> o_scale = l_new / ctx.softmax_l;
    subgroup::tile_broadcast_op<subgroup::tile_div, matOi_t>(matOi, o_scale);

    // update m and l for the next step
    ctx.softmax_m = m_new;
    ctx.softmax_l = l_new;

    if constexpr (kIsTraining) {
      // TODO: apply dropout
    }
  }

  // ==================== // store_Pij // ====================== //

  /// @brief store Pij to local memory.
  inline void store_Pij(matPij_t& matPij) {
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<result_overwrite, gpu_arch::Xe>,
        tile_shape_BmBc,
        mem_desc_Pij_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matPij, ctx.desc_Pij);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
  }

  // ==================== // store_Oi // ====================== //

  /// @brief store Oi to global memory.
  inline void store_Oi(matOi_t& matOi) {
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<result_overwrite, gpu_arch::Xe>,
        tile_shape_BmHm,
        mem_desc_Oi_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matOi, ctx.desc_Oi);
  }

  // ====================== // preload_Qi // ====================== //

  /// @brief preload_Qi is used to load Qi from global to local memory.

  inline void preload_Qi(matQ_t& matQ) {
    using matQi_load_t = subgroup::mem_payload_t<
        scalar_t,
        matQ_tile_desc_t,
        msg_type::block_1d,
        mem_desc_Qi_t::layout,
        mem_desc_Qi_t::space>;

    mem_desc_Qi_t desc_Qi_load(ctx.desc_Qi);

    matQi_load_t matQi_load(desc_Qi_load);
    subgroup::tile_load(matQ, matQi_load);
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_Sij = brgemm_Sij_t::barrier_count;
    constexpr uint32_t barrier_count_Oi = brgemm_Oi_t::barrier_count;
    constexpr uint32_t count =
        std::max(barrier_count_Sij, barrier_count_Oi) + nbarrier_cnt;
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size =
        slm_size_Qi + slm_size_softmax + std::max(slm_size_Pij, slm_size_PV);
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  };

  /// @brief Helper function to get the nd_range under the ifmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<2> get_nd_range(
      uint32_t num_batches,
      uint32_t num_beams,
      uint32_t num_heads) {
    sycl::range<2> local_range = sycl::range<2>{num_beams, wg_size_x};
    sycl::range<2> group_range = sycl::range<2>{num_batches, num_heads};

    return sycl::nd_range<2>{group_range * local_range, local_range};
  };

  // ================= // Entry of the functor // ================= //
  // bias: (b, beams, 1, 1, t)
  // alibi: (b, beams, N, 1, t)
  inline KERNEL_FUNC void load_bias(
      arguments_t& args,
      scalar_t* ptr,
      uint32_t base_offset,
      uint32_t startT,
      uint32_t endT,
      matSij_t& matAcc) {
    base_offset += (startT + ctx.sg_idx * kSgBc);
    constexpr int simd_lanes = kSgBc > 32 ? 32 : 16;

    static_assert(kSgBc % simd_lanes == 0);

    constexpr int loops = kSgBc / simd_lanes;
    xetla_vector<uint32_t, simd_lanes> offsets =
        xetla_vector_gen<uint32_t, simd_lanes>(0, 1);
    offsets *= sizeof(scalar_t);
    offsets += (base_offset * sizeof(scalar_t));
#pragma unroll
    for (int i = 0; i < loops; ++i) {
      offsets += i * simd_lanes * sizeof(scalar_t);
      matAcc.reg.xetla_select<simd_lanes, 1>(i * simd_lanes) =
          xetla_load_global<
              scalar_t,
              1,
              data_size::default_size,
              cache_hint::cached,
              cache_hint::cached,
              simd_lanes>(ptr, offsets);
    }
  }

  inline KERNEL_FUNC void operator()(
      xetla_exec_item<2>& ei,
      arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    ctx.init_context(ei, args);

    matQ_t matQ;
    preload_Qi(matQ);
    // initialize matOi to accumulate the output
    matOi_t matOi(0);

    // iterate through key0 and value0
    for (int32_t start_T = 0; start_T < args.uT0; start_T += kBc) {
      ctx.set_context0(args, start_T);
      // compute Sij
      matSij_t matSij(0);
      gemm0_Sij(matQ, matSij, args);

      prefetch_V0();

      matSij_t matAlibi(0);
      if constexpr (kUseAlibi) {
        load_bias(
            args,
            args.A_ptr,
            ctx.alibi_base_offset,
            start_T,
            args.uT0,
            matAlibi);
      }
      matSij_t matBias(0);
      if constexpr (kUseBias) {
        load_bias(
            args, args.B_ptr, ctx.bias_base_offset, start_T, args.uT0, matBias);
      }

      // softmax
      pre_softmax(matSij, matAlibi, matBias, args, start_T, args.uT0);

      softmax_fwd(matSij, matOi);
      store_Pij(matSij);
      // compute Oi
      gemm0_Oi(matOi, args, start_T);
    }

    // iterate through key1 and value1
    for (int32_t start_T = 0; start_T < args.uT1; start_T += kBc) {
      ctx.set_context1(args, start_T);
      // compute Sij
      matSij_t matSij(0);
      gemm1_Sij(matQ, matSij, args);
      prefetch_v1();

      matSij_t matAlibi(0);
      if constexpr (kUseAlibi) {
        load_bias(
            args,
            args.A_ptr,
            ctx.alibi_base_offset,
            start_T + args.uT0,
            args.uT0 + args.uT1,
            matAlibi);
      }
      matSij_t matBias(0);
      if constexpr (kUseBias) {
        load_bias(
            args,
            args.B_ptr,
            ctx.bias_base_offset,
            start_T + args.uT0,
            args.uT0 + args.uT1,
            matBias);
      }

      // softmax
      pre_softmax(matSij, matAlibi, matBias, args, start_T, args.uT1);
      softmax_fwd(matSij, matOi);
      // compute Oi
      gemm1_Oi(args, matSij, matOi);
    }

    store_Oi(matOi);
  }
}; // ifmha_forward_t
} // namespace fmha
} // namespace gpu::xetla
