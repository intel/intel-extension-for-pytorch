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
    bool kUseBias,
    bool kIsTraining>
class ifmha_forward_t {
 public:
  static_assert((!kIsTraining), "training is not supported yet");

  using accum_t = float;
  using index_t = int32_t;
  static constexpr uint32_t accum_step = ifmha_policy::accum_step;
  static constexpr accum_t kNegInfinity = INFINITY * -1;

  struct arguments_t {
    // Input and output tensors
    scalar_t* Q_ptr; // [1,B,Bm,N,H] - query
    scalar_t* K0_ptr; // [T0,B,1,N,H] - key0
    scalar_t* K1_ptr; // [T1,B,Bm,N,H] - key1
    scalar_t* V0_ptr; // [T0,B,1,N,H] - value0
    scalar_t* V1_ptr; // [T1,B,Bm,N,H] - value1
    index_t* I_ptr; // [T1,B,Bm] - index
    scalar_t* B_ptr = nullptr; // [B,Bm,1,F,T] - bias
    uint8_t* Dp_ptr = nullptr; // [B,Bm,N,F,T] - dropout mask
    scalar_t* O_ptr; // [B,Bm,1,N,H] - output
    // Dropout scale is computed from dropout prob
    accum_t dp_prob;
    accum_t dp_scale;
    // Softmax scale is the reciprocal square root of head size by default
    accum_t sm_scale;
    // Dimension size
    uint32_t uB;
    uint32_t uBm;
    uint32_t uN;
    uint32_t uH;
    uint32_t uT0;
    uint32_t uT1;
    uint32_t uT;

    inline arguments_t(
        scalar_t* query,
        scalar_t* key0,
        scalar_t* key1,
        scalar_t* value0,
        scalar_t* value1,
        int32_t* index,
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
        uint32_t kv_len1)
        : Q_ptr(query),
          K0_ptr(key0),
          K1_ptr(key1),
          V0_ptr(value0),
          V1_ptr(value1),
          I_ptr(index),
          B_ptr(bias),
          Dp_ptr(dropout),
          dp_prob(dropout_prob),
          dp_scale(1.f / (1.f - dropout_prob)),
          sm_scale(sm_scale),
          O_ptr(out),
          uB(num_batches),
          uBm(beam),
          uN(num_heads),
          uH(head_size),
          uT0(kv_len0),
          uT1(kv_len1),
          uT(kv_len0 + kv_len1) {}
  };

 private:
  // ---------------- // Tile shape and Threads // ---------------- //
  static constexpr uint32_t kBc = ifmha_policy::kBc;
  static constexpr uint32_t kHm = ifmha_policy::kHm;
  static constexpr uint32_t kSgBc = ifmha_policy::kSgBc;
  static constexpr uint32_t kSgHm = ifmha_policy::kSgHm;

  using tile_shape_Bc = group::tile_shape_t<kBc, 1, kSgBc, 1>;
  using tile_shape_Hm = group::tile_shape_t<kHm, 1, kSgHm, 1>;
  using work_group_t = typename tile_shape_Bc::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static_assert(
      kHm / kSgHm == kBc / kSgBc,
      "wg_size must be the same between Hm and Bc");
  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  static constexpr uint32_t kHt =
      (kHm + accum_step - 1) / accum_step * accum_step;

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local;
  using mem_desc_Qi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Qi_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Bij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Ot_L_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Oi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Ij_t =
      mem_desc_t<index_t, mem_layout::row_major, mem_space::global>;
  using imem_desc_Kj_t =
      imem_desc_t<scalar_t, kSgBc, mem_layout::row_major, mem_space::global>;
  using imem_desc_Vj_t =
      imem_desc_t<scalar_t, kSgBc, mem_layout::row_major, mem_space::global>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t slm_size_Qi = kHm * sizeof(scalar_t);
  static constexpr uint32_t slm_size_Ot = kHt * wg_size * sizeof(accum_t);
  static constexpr uint32_t slm_size_softmax =
      (wg_size > 1) ? wg_size * sizeof(accum_t) : 0;
  // Slm addr to store inermediate results
  static constexpr uint32_t Qi_slm = 0;
  static constexpr uint32_t Ot_slm = Qi_slm + slm_size_Qi;
  static constexpr uint32_t softmax_slm = Ot_slm + slm_size_Ot;

  static constexpr uint32_t nbarrier_cnt = (wg_size > 1) ? 1 : 0;

  // ======================== // Context // ======================= //

  /// @brief Used to store variables in the ifmha loops
  struct context_t {
    // thread id
    uint32_t sg_id;
    uint32_t loop_count;
    // nbarrier
    xetla_nbarrier_t<wg_size, wg_size> nbarrier;
    // softmax statistics
    xetla_vector<accum_t, 1> softmax_m;
    xetla_vector<accum_t, 1> softmax_l;
    // mem desc variables
    mem_desc_Qi_t desc_Qi;
    mem_desc_Qi_L_t desc_Qi_L;
    mem_desc_Bij_t desc_Bij;
    mem_desc_Ot_L_t desc_Ot0_L;
    mem_desc_Ot_L_t desc_Ot1_L;
    mem_desc_Oi_t desc_Oi;
    mem_desc_Ij_t desc_Ij;
    imem_desc_Kj_t idesc_Kj;
    imem_desc_Vj_t idesc_Vj;

    inline context_t() = default;

    /// @brief Initialize invariant variables in the ifmha loop
    inline void init_context(xetla_exec_item<2>& ei, arguments_t& args) {
      // thread id
      sg_id = ei.get_local_linear_id();
      // loop_count of gemm_sij
      loop_count = (args.uH + accum_step - 1) / accum_step;
      // nbarrier
      nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
      // softmax statistics
      softmax_m = kNegInfinity;
      softmax_l = 0.f;

      // mem desc variables
      uint32_t gid = ei.get_group(0);
      int32_t start_y = gid;
      uint32_t end_y = start_y + 1;

      desc_Qi.init(args.Q_ptr, {args.uH, end_y, args.uH}, {0, start_y});
      desc_Oi.init(args.O_ptr, {args.uH, end_y, args.uH}, {0, start_y});
      idesc_Kj.init(args.K0_ptr, args.K1_ptr, args.uH);
      idesc_Vj.init(args.V0_ptr, args.V1_ptr, args.uH);

      desc_Qi_L.init(Qi_slm, {kHm, 1, kHm}, {0, 0});
      desc_Ot0_L.init(Ot_slm, {kHt, wg_size, kHt}, {0, int(sg_id)});
      desc_Ot1_L.init(Ot_slm, {kHt, wg_size, kHt}, {int(sg_id * kSgHm), 0});
    }

    inline void load_index(
        xetla_exec_item<2>& ei,
        arguments_t& args,
        xetla_vector<int32_t, kSgBc>& index,
        int32_t start_y) {
      using index_tile_desc_t = subgroup::tile_desc_t<1, kSgBc, 1, kSgBc>;
      using index_tile_t = subgroup::tile_t<index_t, index_tile_desc_t>;
      using index_payload_t = subgroup::mem_payload_t<
          index_t,
          index_tile_desc_t,
          msg_type::block_2d,
          mem_desc_Ij_t::layout,
          mem_desc_Ij_t::space>;
      index_tile_t index_tile;
      index_payload_t index_payload;

      auto gid = ei.get_group(0);
      auto b_bm_id = gid / args.uN;
      int32_t start_x = b_bm_id;
      uint32_t end_x = start_x + 1;
      uint32_t end_y = std::min(start_y + kSgBc, args.uT1);
      desc_Ij.init(
          args.I_ptr, {end_x, end_y, args.uB * args.uBm}, {start_x, start_y});
      index_payload.init(desc_Ij);
      subgroup::tile_load(index_tile, index_payload);
      index = index_tile.reg;
    }

    /// @brief Update variables for each ifmha loop
    inline void update_context(
        xetla_exec_item<2>& ei,
        arguments_t& args,
        uint32_t start_T) {
      uint32_t start_T_sg = start_T + sg_id * kSgBc;
      uint32_t end_T_sg = std::min(start_T_sg + kSgBc, args.uT);

      auto gid = ei.get_group(0);
      auto nid = gid % args.uN;
      auto bid = gid / (args.uBm * args.uN);

      // compute index for buffer0
      uint32_t end_T_0 = std::min(end_T_sg, args.uT0);
      uint32_t lanes0 = std::max(end_T_0 - start_T_sg, uint32_t(0));
      xetla_vector<int32_t, kSgBc> index0;
      if (lanes0 > 0) {
        int32_t offset = start_T_sg * args.uB * args.uN + bid * args.uN + nid;
        index0 = xetla_vector_gen<int32_t, kSgBc>(offset, args.uB * args.uN);
      }

      // compute index for buffer1
      uint32_t start_T_1 = start_T_sg > args.uT0 ? start_T_sg - args.uT0 : 0;
      uint32_t end_T_1 = end_T_sg > args.uT0 ? end_T_sg - args.uT0 : 0;
      uint32_t lanes1 = end_T_1 - start_T_1;
      xetla_vector<int32_t, kSgBc> index1;
      if (lanes1 > 0) {
        load_index(ei, args, index1, start_T_1);
        index1 = index1 * args.uB * args.uBm * args.uN + gid;
      }

      idesc_Kj.update_index(index0, index1, lanes0, lanes1);
      idesc_Vj.update_index(index0, index1, lanes0, lanes1);

      if constexpr (kUseBias) {
        // TODO
      }
    }
  };

  context_t ctx;

  // ======================= // gemm_Sij // ======================= //

  using matSij_tile_desc_t = subgroup::tile_desc_t<kSgBc, 1, kSgBc, 1>;
  using matSij_t = subgroup::tile_t<accum_t, matSij_tile_desc_t>;

  /// @brief gemm_Sij is used to compute Sij = Qi x Kj.T
  inline void gemm_Sij(matSij_t& matSij, arguments_t& args) {
    using matQi_tile_desc_t =
        subgroup::tile_desc_t<accum_step, 1, accum_step, 1>;
    using matQi_payload_t = subgroup::mem_payload_t<
        scalar_t,
        matQi_tile_desc_t,
        msg_type::block_1d,
        mem_layout::row_major,
        mem_space::local>;
    using matQi_load_t = subgroup::tile_t<scalar_t, matQi_tile_desc_t>;
    using matQi_t = subgroup::tile_t<accum_t, matQi_tile_desc_t>;
    // TODO: how to set block_size_x/y
    using matKj_tile_desc_t =
        subgroup::tile_desc_t<accum_step, kSgBc, accum_step, kSgBc>;
    using matKj_t = subgroup::tile_t<accum_t, matKj_tile_desc_t>;

    // Gemm to comput Sij
    matQi_payload_t matQi_payload(ctx.desc_Qi_L);
    matQi_load_t matQi_load;
    matQi_t matQi;
    matKj_t matKj;

    for (int i = 0; i < ctx.loop_count; i++) {
      // load Qi from local memory
      subgroup::tile_load(matQi_load, matQi_payload);
      subgroup::elemwise_cvt(matQi, matQi_load);
      matQi_payload.template update_tdesc<tdesc_update_dir::x_dir>(accum_step);

      // indexed load Kj
      iload_tile(matKj, ctx.idesc_Kj);
      ctx.idesc_Kj.update_offset(i * accum_step);

#pragma unroll
      for (int i = 0; i < kSgBc; i++) {
        auto matKj_sub = matKj.reg.xetla_select<accum_step, 1>(i * accum_step);
        matSij.reg.xetla_select<1, 1>(i) +=
            xetla_reduce<accum_t, accum_t, accum_step, reduce_op::sum>(
                matQi.reg * matKj_sub);
      }
    }

    // Multiply by softmax scaling factor
    matSij.reg *= args.sm_scale;

    if constexpr (kUseBias) {
      // TODO:
    }
  }

  // ======================= // gemm_Oi // ======================= //

  using matOi_tile_desc_t = subgroup::tile_desc_t<kSgHm, 1, kSgHm, 1>;
  using matOi_t = subgroup::tile_t<accum_t, matOi_tile_desc_t>;
  using matPij_t = matSij_t;

  /// @brief gemm_Oi is used to compute Oi += Pij x Vj
  inline void gemm_Oi(matPij_t& matPij, matOi_t& matOi, arguments_t& args) {
    using matVj_tile_desc_t =
        subgroup::tile_desc_t<accum_step, kSgBc, accum_step, kSgBc>;
    using matVj_t = subgroup::tile_t<accum_t, matVj_tile_desc_t>;

    using matOt0_tile_desc_t =
        subgroup::tile_desc_t<accum_step, 1, accum_step, 1>;
    using matOt0_payload_t = subgroup::mem_payload_t<
        accum_t,
        matOt0_tile_desc_t,
        msg_type::block_1d,
        mem_layout::row_major,
        mem_space::local>;
    using matOt0_t = subgroup::tile_t<accum_t, matOt0_tile_desc_t>;

    matVj_t matVj;
    matOt0_t matOt0;
    matOt0_payload_t matOt0_payload(ctx.desc_Ot0_L);

    for (int i = 0; i < ctx.loop_count; i++) {
      // indexed load Vj
      iload_tile(matVj, ctx.idesc_Vj);
      ctx.idesc_Vj.update_offset(i * accum_step);

#pragma unroll
      for (int j = 0; j < accum_step; j++) {
        auto matVj_sub = matVj.reg.xetla_select<kSgBc, accum_step>(j);
        matOt0.reg.xetla_select<1, 1>(j) +=
            xetla_reduce<accum_t, accum_t, kSgBc, reduce_op::sum>(
                matPij.reg * matVj_sub);
      }

      subgroup::tile_store(matOt0, matOt0_payload);
      matOt0_payload.template update_tdesc<tdesc_update_dir::x_dir>(accum_step);
    }

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();

    using matOt1_payload_t = subgroup::mem_payload_t<
        accum_t,
        matOi_tile_desc_t,
        msg_type::block_1d,
        mem_layout::row_major,
        mem_space::local>;
    using matOt1_t = subgroup::tile_t<accum_t, matOi_tile_desc_t>;

    matOt1_t matOt1;
    matOt1_payload_t matOt1_payload(ctx.desc_Ot1_L);

#pragma unroll
    for (int i = 0; i < wg_size; i++) {
      subgroup::tile_load(matOt1, matOt1_payload);
      matOt1_payload.template update_tdesc<tdesc_update_dir::y_dir>(1);
      matOi.reg += matOt1.reg;
    }
  }

  // ====================== // softmax_fwd // ===================== //

  using wg_max_t = group_1d_reduce_t<matSij_t, wg_size, reduce_op::max>;
  using wg_sum_t = group_1d_reduce_t<matSij_t, wg_size, reduce_op::sum>;

  /// @brief softmax_fwd is used to do softmax.
  inline void softmax_fwd(
      matSij_t& matSij,
      matOi_t& matOi,
      arguments_t& args,
      uint32_t start_T) {
    // padding mask the tail, if needed.
    uint32_t sg_start_T = start_T + ctx.sg_id * kSgBc;
    uint32_t num_keep = (args.uT < sg_start_T) ? 0 : (args.uT - sg_start_T);
    if (num_keep < kSgBc) {
      xetla_mask<kSgBc> mask =
          xetla_vector_gen<uint32_t, kSgBc>(1, 1) > num_keep;
      matSij.reg.xetla_merge(kNegInfinity, mask);
    }

    // compute new m
    wg_max_t wg_max(ctx.sg_id, 0, softmax_slm);
    xetla_vector<accum_t, 1> m_new = wg_max(matSij);
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive();
    m_new = xetla_max<accum_t, 1>(m_new, ctx.softmax_m);

    // correct old l
    ctx.softmax_l *= xetla_exp<accum_t, 1>(ctx.softmax_m - m_new);
    // compute Pij
    subgroup::tile_broadcast_op<subgroup::tile_minus, matSij_t>(matSij, m_new);
    matSij.reg = xetla_exp<accum_t>(matSij.reg);

    // compute new l
    if constexpr (wg_size > 1)
      ctx.nbarrier.wait();
    wg_sum_t wg_sum(ctx.sg_id, 0, softmax_slm);
    xetla_vector<accum_t, 1> l_new = wg_sum(matSij);
    l_new += ctx.softmax_l;

    // rescale operands of matmuls
    subgroup::tile_broadcast_op<subgroup::tile_div, matSij_t>(matSij, l_new);
    xetla_vector<accum_t, 1> o_scale = l_new / ctx.softmax_l;
    subgroup::tile_broadcast_op<subgroup::tile_div, matOi_t>(matOi, o_scale);
    // update m and l for the next step
    ctx.softmax_m = m_new;
    ctx.softmax_l = l_new;

    if constexpr (kIsTraining) {
      // TODO: apply dropout
    }
  }

  // ==================== // store_Oi // ====================== //

  /// @brief store Oi to global memory. [B,N,F,H]
  inline void store_Oi(matOi_t& matOi, arguments_t& args) {
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape_Hm,
        mem_desc_Oi_t>;
    epilogue_t epilogue;
    work_group_t g(ctx.sg_id);
    epilogue(g, matOi, ctx.desc_Oi);
  }

  // ====================== // preload_Qi // ====================== //

  /// @brief preload_Qi is used to load Qi from global to local memory.
  inline void preload_Qi(arguments_t& args) {
    using matQi_t = subgroup::tile_t<scalar_t, matOi_tile_desc_t>;
    using matQi_load_t = subgroup::mem_payload_t<
        scalar_t,
        matOi_tile_desc_t,
        msg_type::block_2d,
        mem_desc_Qi_t::layout,
        mem_desc_Qi_t::space,
        gpu_arch::Xe>;
    using matQi_store_t = subgroup::mem_payload_t<
        scalar_t,
        matOi_tile_desc_t,
        subgroup::msg_type_v<matOi_tile_desc_t, mem_desc_Qi_L_t::space>,
        mem_desc_Qi_L_t::layout,
        mem_desc_Qi_L_t::space,
        gpu_arch::Xe>;

    mem_desc_Qi_t desc_Qi_load(ctx.desc_Qi);
    mem_desc_Qi_L_t desc_Qi_store(ctx.desc_Qi_L);

    int32_t tile_offset_x = ctx.sg_id * kSgHm;
    desc_Qi_load.update_coord_x(tile_offset_x);
    desc_Qi_store.update_coord_x(tile_offset_x);

    matQi_t matQi;
    matQi_load_t matQi_load(desc_Qi_load);
    subgroup::tile_load(matQi, matQi_load);

    matQi_store_t matQi_store(desc_Qi_store);
    subgroup::tile_store(matQi, matQi_store);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size > 1)
      ctx.nbarrier.arrive_wait();
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    return nbarrier_cnt;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_Qi + slm_size_Ot + slm_size_softmax;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  };

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<2> get_nd_range(uint32_t total_batches) {
    // local range
    sycl::range<2> local_range = sycl::range<2>{1, wg_size};
    // group range
    sycl::range<2> group_range = sycl::range<2>{total_batches, 1};
    return sycl::nd_range<2>{group_range * local_range, local_range};
  };

  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(
      xetla_exec_item<2>& ei,
      arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context for ifmha loops
    ctx.init_context(ei, args);
    // preload Qi to local memory
    preload_Qi(args);
    // initialize matOi for accumulate the output
    matOi_t matOi(0);

    // iterate through the keys
    for (uint32_t start_T = 0; start_T < args.uT; start_T += kBc) {
      // update context for current loop
      ctx.update_context(ei, args, start_T);
      // compute Sij
      matSij_t matSij(0);
      gemm_Sij(matSij, args);
      // softmax
      softmax_fwd(matSij, matOi, args, start_T);
      // compute Oi
      gemm_Oi(matSij, matOi, args);
    }

    // Store output to global
    store_Oi(matOi, args);
  }
}; // ifmha_forward_t

} // namespace fmha
} // namespace gpu::xetla
