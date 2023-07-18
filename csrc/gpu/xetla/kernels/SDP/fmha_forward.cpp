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
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/

#pragma once

//#include <ATen/record_function.h>
#include <utils/DPCPP.h>
#include <limits>
#include "../../mha.h"
#include "fmha_policy.h"

// Set to 1 to get raw output, not permuted
#define _RAW_OUTPUT 1

// namespace xpu {
namespace gpu::xetla {

namespace {

template <
    typename fmha_policy,
    typename scalar_t,
    bool kUseBias,
    bool kUseDropout>
class fmha_forward_t {
  static_assert((!kUseDropout), "dropout not supported yet");

  using accum_t = float;
  static constexpr uint32_t SIMD = fmha_policy::SIMD;
  static constexpr uint32_t accum_step = fmha_policy::accum_step;
  static constexpr uint32_t stages = fmha_policy::stages;
  static constexpr uint32_t sync_freq = fmha_policy::sync_freq;

  static constexpr uint32_t kQueryBlock = fmha_policy::wg_tile0_m;
  static constexpr uint32_t kKeyBlock = fmha_policy::wg_tile0_n;
  static constexpr uint32_t kMaxHeadSize = fmha_policy::wg_tile1_n;

  static_assert(
      kQueryBlock == fmha_policy::wg_tile1_m,
      "wg_tile0_m should be the same as wg_tile1_m");
  static_assert(
      kKeyBlock % SIMD == 0,
      "wg_tile0_n should be a multiple of SIMD");
  static_assert(
      kMaxHeadSize % SIMD == 0,
      "wg_tile1_n should be a multiple of SIMD");

  // local memory size
  static constexpr uint32_t slm_size_score =
      kQueryBlock * kKeyBlock * sizeof(accum_t);
  static constexpr uint32_t slm_size_ml = 3 * kQueryBlock * sizeof(accum_t);

  // Slm addr to store inermediate results
  // score is the result of Qi x Kj.T as well as subsequent softmax results
  static constexpr uint32_t score_base = 0;
  // ml is the row-max (m) and row-exp-sum of score (l)
  static constexpr uint32_t ml_base = score_base + slm_size_score;

 public:
  struct arguments_t {
    // Input tensors
    scalar_t* query_ptr; // [B, N, F, H]
    scalar_t* key_ptr; // [B, N, T, H]
    scalar_t* value_ptr; // [B, N, T, H]
    scalar_t* bias_ptr = nullptr; // [B, 1, F, T]
    uint8_t* dp_mask_ptr = nullptr; // [B, N, F, T]
    accum_t dp_prob;
    accum_t scale;

    // Output tensor
    scalar_t* out_ptr; // [B, F, N, H]

    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;
    // reciprocal of head size
    accum_t fH_rcp;

    // Offsets and boundaries for flash mha loop
    int startF;
    int startT;
    uint32_t boundaryF;
    uint32_t boundaryT;

    int startF_bias;
    uint32_t boundaryF_bias;

    // Constructors
    inline arguments_t() = default;
    inline arguments_t(
        scalar_t* query,
        scalar_t* key,
        scalar_t* value,
        scalar_t* bias,
        uint8_t* dropout_mask,
        accum_t dropout_prob,
        scalar_t* out,
        uint32_t num_batches,
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t num_queries,
        uint32_t num_keys)
        : query_ptr(query),
          key_ptr(key),
          value_ptr(value),
          bias_ptr(bias),
          dp_mask_ptr(dropout_mask),
          dp_prob(dropout_prob),
          scale(1.f / (1.f - dropout_prob)),
          out_ptr(out),
          uB(num_batches),
          uN(num_heads),
          uH(head_size),
          uF(num_queries),
          uT(num_keys),
          fH_rcp(xetla_rsqrt<accum_t>(accum_t(head_size))) {}

    // Moves coords to what we should process
    inline void init_coords(xetla_exec_item<3>& ei) {
      uint32_t gid = ei.get_group(0);
      uint32_t block_id = ei.get_group(1);
      // query (F)
      startF = gid * uF + block_id * kQueryBlock;
      boundaryF = (gid + 1) * uF;
      // key (T)
      startT = gid * uT;
      boundaryT = (gid + 1) * uT;

      if constexpr (kUseBias) {
        uint32_t batch_id = gid / uN;
        startF_bias = batch_id * uF + block_id * kQueryBlock;
        boundaryF = (batch_id + 1) * uF;
      }
    }
    inline void update_coords() {
      startT += kKeyBlock;
    }
  };

 private:
  struct gemm0_t {
    /*
      This functor is used to compute a block of score
      query x key.T = score  # [bF,H] x [H,bT] = [bF,bT]
    */

    // tile description
    static constexpr uint32_t wg_tile_m = fmha_policy::wg_tile0_m;
    static constexpr uint32_t wg_tile_n = fmha_policy::wg_tile0_n;
    static constexpr uint32_t sg_tile_m = fmha_policy::sg_tile0_m;
    static constexpr uint32_t sg_tile_n = fmha_policy::sg_tile0_n;
    using tile_shape = gpu::xetla::group::
        tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

    // Number of threads used in gemm0
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t wg_size = wg_size_x * wg_size_y;

    // Using brgemm_selector to get a specific brgemm class
    using brgemm_t = typename gpu::xetla::group::brgemm_selector_t<
        scalar_t,
        scalar_t,
        mem_layout::row_major,
        mem_layout::col_major,
        mem_space::global,
        mem_space::global,
        8,
        8,
        accum_t,
        tile_shape,
        accum_step,
        mma_engine::xmx,
        gpu_arch::Xe,
        stages,
        sync_freq>::brgemm;

    // Add bias if needed
    using bias_op_t =
        subgroup::elemwise_reduce_op_t<reduce_op::sum, scalar_t, gpu_arch::Xe>;

    // Using epilogue function to save results
    using epilogue_t = gpu::xetla::group::epilogue_t<
        gpu::xetla::group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape,
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>>;

    // Alias
    using brgemm_args_t = typename brgemm_t::arguments_t;
    using work_group_t = typename brgemm_t::work_group_t;
    using mem_desc_a_t = typename brgemm_t::mem_desc_a_t;
    using mem_desc_b_t = typename brgemm_t::mem_desc_b_t;
    using matAcc_t = typename brgemm_t::matAcc_t;
    using bias_args_t = typename bias_op_t::arguments_t;
    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;

    // functor call
    inline KERNEL_FUNC void operator()(arguments_t* args, uint32_t sg_id) {
      if (sg_id >= wg_size)
        return;

      // compute wrok group level boundaries
      uint32_t boundary_m = (args->startF + wg_tile_m) > args->boundaryF
          ? args->boundaryF
          : (args->startF + wg_tile_m);
      uint32_t boundary_n = (args->startT + wg_tile_n) > args->boundaryT
          ? args->boundaryT
          : (args->startT + wg_tile_n);

      // query x key.T = score  # [bF,H] x [H,bT] = [bF,bT]
      mem_desc_a_t mem_desc_a(
          args->query_ptr, {args->uH, boundary_m, args->uH}, {0, args->startF});
      mem_desc_b_t mem_desc_b(
          args->key_ptr, {boundary_n, args->uH, args->uH}, {args->startT, 0});
      uint32_t inner_loop_count = (args->uH + accum_step - 1) / accum_step;

      brgemm_t brgemm;
      work_group_t g(sg_id);
      matAcc_t matAcc(0);
      brgemm_args_t brgemm_args(mem_desc_a, mem_desc_b, inner_loop_count);
      brgemm(g, matAcc, brgemm_args, 0, /* nbarrier_base */ 1);

      // Multiply by scaling factor
      matAcc.reg *= args->fH_rcp;

      // Add bias if needed
      if constexpr (kUseBias) {
        bias_op_t bias_op;
        bias_args_t bias_args(
            args->bias_ptr, {args->uT, args->boundaryF_bias, args->uT});
        bias_op(matAcc, /* coord */ {args->startT, args->startF_bias});
      }

      // Do save results to local memory
      epilogue_t epilogue_op;
      mem_desc_c_t mem_desc_c(
          score_base, {wg_tile_n, wg_tile_m, wg_tile_n}, {0, 0});
      epilogue_op(g, matAcc, mem_desc_c);
    }
  };

  struct gemm1_t {
    /*
      This functor is used to compute output and add to acc
      1. acc *= lscale
      2. score x value = output  # [bF,bT] x [bT,H] = [bF,H]
      3. acc += output
    */

    // tile description
    static constexpr uint32_t wg_tile_m = fmha_policy::wg_tile1_m;
    static constexpr uint32_t wg_tile_n = fmha_policy::wg_tile1_n;
    static constexpr uint32_t sg_tile_m = fmha_policy::sg_tile1_m;
    static constexpr uint32_t sg_tile_n = fmha_policy::sg_tile1_n;
    using tile_shape = gpu::xetla::group::
        tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

    // Number of threads used in gemm0
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t wg_size = wg_size_x * wg_size_y;

    // compute policy
    using compute_attr =
        gpu::xetla::group::compute_attr_t<scalar_t, scalar_t, accum_t>;
    using perf_tuning_knob =
        gpu::xetla::group::perf_tuning_knob_t<accum_step, stages, sync_freq>;
    using compute_policy = gpu::xetla::group::compute_policy_default_xmx<
        compute_attr,
        perf_tuning_knob,
        gpu_arch::Xe>;
    // memory layout
    using mem_desc_a_t =
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::local>;
    using mem_desc_b_t =
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;

    using brgemm_t = gpu::xetla::group::
        brgemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

    // epilogue to store raw output, not permuted
    using mem_desc_output_c =
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
    using epilogue_t = gpu::xetla::group::epilogue_t<
        gpu::xetla::group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape,
        mem_desc_output_c>;

    // Alias
    using brgemm_args_t = typename brgemm_t::arguments_t;
    using work_group_t = typename brgemm_t::work_group_t;
    using matAcc_t = typename brgemm_t::matAcc_t;

    // functor call
    inline KERNEL_FUNC void operator()(
        arguments_t* args,
        matAcc_t& outAcc,
        uint32_t sg_id) {
      if (sg_id >= wg_size)
        return;

      // Multiply the outAcc by the lscale from local memory
      xetla_vector<accum_t, 3> v_ml;
      uint32_t j = sg_id / wg_size_x * sg_tile_m;

      for (int i = 0; i < sg_tile_m; ++i, ++j) {
        // Get lscale from local memory
        uint32_t offset_ml = ml_base + j * 3 * sizeof(accum_t);
        v_ml = xetla_load_local<accum_t, 3>(offset_ml);
        accum_t lscale = v_ml[2];

        auto v_acc = outAcc.reg.xetla_select<sg_tile_n, 1>(i * sg_tile_n);
        v_acc *= lscale;
      }

      // compute wrok group level boundaries
      uint32_t boundary_k = (args->startT + kKeyBlock) > args->boundaryT
          ? args->boundaryT
          : (args->startT + kKeyBlock);

      // score x value = output  # [bF,bT] x [bT,H] = [bF,H]
      mem_desc_a_t mem_desc_a(
          score_base, {kKeyBlock, kQueryBlock, kKeyBlock}, {0, 0});
      mem_desc_b_t mem_desc_b(
          args->value_ptr, {args->uH, boundary_k, args->uH}, {0, args->startT});
      uint32_t inner_loop_count =
          (boundary_k - args->startT + accum_step - 1) / accum_step;

      brgemm_t brgemm;
      work_group_t g(sg_id);
      brgemm_args_t brgemm_args(mem_desc_a, mem_desc_b, inner_loop_count);
      brgemm(g, outAcc, brgemm_args, 0, /* nbarrier_base */ 1);
    }

    inline KERNEL_FUNC void raw_store(
        xetla_exec_item<3>& ei,
        arguments_t* args,
        matAcc_t& outAcc) {
      uint32_t sg_id = ei.get_local_linear_id();
      if (sg_id >= wg_size)
        return;

      uint32_t boundary_m = (args->startF + wg_tile_m) > args->boundaryF
          ? args->boundaryF
          : (args->startF + wg_tile_m);
      mem_desc_output_c md_c(
          args->out_ptr, {args->uH, boundary_m, args->uH}, {0, args->startF});
      epilogue_t epilogue;
      work_group_t g(sg_id);
      epilogue(g, outAcc, md_c);
    }

    // permute store outAcc [b,n,F,H] to output [B,F,N,H]
    inline KERNEL_FUNC void permute_store(
        xetla_exec_item<3>& ei,
        arguments_t* args,
        matAcc_t& outAcc) {
      uint32_t sg_id = ei.get_local_linear_id();
      if (sg_id >= wg_size)
        return;

      // start point of current work group
      uint32_t sg_idx = sg_id % wg_size_x;
      uint32_t sg_idy = sg_id / wg_size_x;

      uint32_t b = ei.get_group(0) / args->uN;
      uint32_t n = ei.get_group(0) % args->uN;
      uint32_t f = sg_idy * sg_tile_m + ei.get_group(1) * wg_tile_m;
      uint32_t h = sg_idx * sg_tile_n;

      if (h >= args->uH)
        return;

      xetla_tdescriptor transpose_tdecs;
      xetla_vector<scalar_t, sg_tile_n> v_out;

      uint32_t height = args->uB * args->uN * args->uF;
      uint32_t offset_height = b * args->uN * args->uF + f * args->uN + n;

      xetla_fill_tdesc<scalar_t, sg_tile_n, 1, 1>(
          transpose_tdecs.xetla_format<uint32_t>(),
          args->out_ptr,
          args->uH,
          height,
          args->uH,
          h,
          offset_height);

      for (uint32_t i = 0; i < sg_tile_m && f < args->uF; ++i, ++f) {
        // load data from outAcc
        auto v_acc = outAcc.reg.xetla_select<sg_tile_n, 1>(i * sg_tile_n);
        v_out = xetla_cvt<scalar_t, accum_t, sg_tile_n>(v_acc);

        xetla_tstore_global<
            scalar_t,
            sg_tile_n,
            cache_hint::write_back,
            cache_hint::write_back>(transpose_tdecs, v_out);
        xetla_update_tdesc_offsety(
            transpose_tdecs.xetla_format<uint32_t>(), args->uN);
      }
    }
  };

 public:
  static constexpr uint32_t kThreadNum =
      gemm0_t::wg_size > gemm1_t::wg_size ? gemm0_t::wg_size : gemm1_t::wg_size;
  static_assert(
      kThreadNum <= 32,
      "The number of threads should be less than 32!");

  /// @brief Helper function to get the local range under the Fmha policy.
  /// @return Expected local range.
  static sycl::range<3> get_local_range() {
    if constexpr (gemm0_t::wg_size > gemm1_t::wg_size) {
      constexpr uint32_t local_range_m = gemm0_t::wg_size_y;
      constexpr uint32_t local_range_n = gemm0_t::wg_size_x;
      return sycl::range<3>{1, local_range_m, local_range_n};
    } else {
      constexpr uint32_t local_range_m = gemm1_t::wg_size_y;
      constexpr uint32_t local_range_n = gemm1_t::wg_size_x;
      return sycl::range<3>{1, local_range_m, local_range_n};
    }
  };

  /// @brief Helper function to get the group range under the Fmha policy.
  /// @return Expected group range.
  static sycl::range<3> get_group_range(
      uint32_t total_batches,
      uint32_t num_queries) {
    uint32_t group_range_m = (num_queries + kQueryBlock - 1) / kQueryBlock;
    return sycl::range<3>{total_batches, group_range_m, 1};
  };

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(
      uint32_t total_batches,
      uint32_t num_queries) {
    sycl::range<3> local_range = get_local_range();
    sycl::range<3> group_range = get_group_range(total_batches, num_queries);
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

 private:
  struct softmax_t {
    /*
    This functor is used to compute fmha softmax
    1. load score from local memory [kQueryBlock, kKeyBlock]
    2. compute softmax for each line
    */

    static constexpr uint32_t kScoreHeight = kKeyBlock / SIMD;

    // tiles used to load and store score data
    using score_tile_desc_t = subgroup::
        tile_desc_t<SIMD, kScoreHeight, SIMD, kScoreHeight, reg_layout::tiled>;
    using score_rw_t = subgroup::tile_t<accum_t, score_tile_desc_t>;
    using score_rw_payload_t = subgroup::mem_payload_t<
        accum_t,
        score_tile_desc_t,
        subgroup::msg_type_v<score_tile_desc_t, mem_space::local>,
        mem_layout::row_major,
        mem_space::local,
        gpu_arch::Xe>;

    // tiles of the acc data
    using matAcc_t = typename gemm1_t::matAcc_t;

    // functor call
    template <bool kIsFirst, bool kHasTail>
    inline KERNEL_FUNC void call(
        arguments_t* args,
        matAcc_t& outAcc,
        uint32_t sg_id,
        uint32_t remainT) {
      // params for loading/storing score
      uint32_t score_offset_y = kScoreHeight * sg_id;
      constexpr uint32_t score_boundary = kScoreHeight * kQueryBlock;
      constexpr int score_stride = kScoreHeight * kThreadNum;

      score_rw_t score_rw;
      score_rw_payload_t score_rw_payload(
          score_base, SIMD, score_boundary, SIMD, 0, score_offset_y);

      xetla_vector<accum_t, kKeyBlock> v_score;
      xetla_vector<accum_t, 3> v_ml;
      xetla_mask<kKeyBlock> v_mask;

      if constexpr (kHasTail) {
        v_mask = xetla_vector_gen<uint32_t, kKeyBlock>(1, 1) > remainT;
      }

      for (uint32_t row = sg_id; row < kQueryBlock; row += kThreadNum) {
        // Load score from local memory
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
            score_rw, score_rw_payload);
        v_score = score_rw.reg.xetla_select<kKeyBlock, 1>(0);

        // Get m_prev and l_prev from local memory
        uint32_t offset_ml = ml_base + row * 3 * sizeof(accum_t);
        v_ml = xetla_load_local<accum_t, 3>(offset_ml);
        accum_t m_prev = v_ml[0];
        accum_t l_prev = v_ml[1];

        // compute new max score (m_curr)
        if constexpr (kHasTail) {
          v_score.xetla_merge(std::numeric_limits<accum_t>::lowest(), v_mask);
        }
        accum_t m_curr =
            xetla_reduce<accum_t, accum_t, kKeyBlock, reduce_op::max>(v_score);
        if constexpr (!kIsFirst) {
          m_curr = xetla_max<accum_t>(m_curr, m_prev);
        }

        // correct l_prev
        if constexpr (kIsFirst) {
          l_prev = 0.f;
        } else {
          l_prev *= xetla_exp<accum_t>(m_prev - m_curr);
        }

        // compute attention weights and new l_curr
        v_score = xetla_exp<accum_t, kKeyBlock>(v_score - m_curr);
        if constexpr (kHasTail) {
          v_score.xetla_merge(0.f, v_mask);
        }
        accum_t l_curr = l_prev +
            xetla_reduce<accum_t, accum_t, kKeyBlock, reduce_op::sum>(v_score);

        // rescale operands of matmuls
        accum_t l_rcp = 1.f / l_curr;
        v_score *= l_rcp;

        // store m_curr, l_curr and lscale to local memory
        v_ml[0] = m_curr;
        v_ml[1] = l_curr;
        v_ml[2] = l_prev * l_rcp;
        xetla_store_local<accum_t, 3>(offset_ml, v_ml);

        // store softmax results back to local memory
        score_rw.reg.xetla_select<kKeyBlock, 1>(0) =
            xetla_cvt<accum_t, accum_t, kKeyBlock>(v_score);
        subgroup::tile_store(score_rw, score_rw_payload);
        score_rw_payload.template update_tdesc<tdesc_update_dir::y_dir>(
            score_stride);
      }
    }
  };

  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count0 = gemm0_t::brgemm_t::barrier_count;
    constexpr uint32_t barrier_count1 = gemm1_t::brgemm_t::barrier_count;
    constexpr uint32_t count = std::max(barrier_count0, barrier_count1) + 1;
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_score + slm_size_ml;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  };

 public:
  // Entry of the fmha forward kernel
  inline KERNEL_FUNC void operator()(
      xetla_exec_item<3>& ei,
      arguments_t* args) {
    // RECORD_FUNCTION("fsdp_forward_no_mask_no_casual_no_stride", {});
    // Moves offsets to what we should process
    args->init_coords(ei);
    uint32_t sg_id = ei.get_local_linear_id();

    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // One barrier is needed to do thread sync while store results to SLM
    xetla_nbarrier_t<kThreadNum, kThreadNum> nbarrier;
    nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);

    // instantiate ops
    gemm0_t gemm0_op;
    gemm1_t gemm1_op;
    softmax_t softmax_op;

    // initialize outAcc for accumulate the output
    typename gemm1_t::matAcc_t outAcc(0);

    gemm0_op(args, sg_id);
    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    uint32_t remainT = args->uT;
    if (remainT < kKeyBlock) {
      softmax_op.template call<true, true>(args, outAcc, sg_id, remainT);
    } else {
      softmax_op.template call<true, false>(args, outAcc, sg_id, remainT);
    }
    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    gemm1_op(args, outAcc, sg_id);

    // Iterate through the rest keys
    for (int offsetT = kKeyBlock; offsetT < args->uT; offsetT += kKeyBlock) {
      xetla_fence<memory_kind::shared_local>();
      nbarrier.arrive_wait();

      args->update_coords();

      gemm0_op(args, sg_id);
      xetla_fence<memory_kind::shared_local>();
      nbarrier.arrive_wait();

      remainT = args->uT - offsetT;
      if (remainT < kKeyBlock) {
        softmax_op.template call<false, true>(args, outAcc, sg_id, remainT);
      } else {
        softmax_op.template call<false, false>(args, outAcc, sg_id, remainT);
      }
      xetla_fence<memory_kind::shared_local>();
      nbarrier.arrive_wait();

      gemm1_op(args, outAcc, sg_id);
    }

    // Store output to global
#if _RAW_OUTPUT
    gemm1_op.raw_store(ei, args, outAcc);
#else
    gemm1_op.permute_store(ei, args, outAcc);
#endif
  }
};

template <typename fmha_policy, typename T, bool kUseBias, bool kUseDropout>
class FmhaForwardKernel;

// The launcher of fmha forward kernel
template <typename fmha_policy, typename T, bool kUseBias, bool kUseDropout>
void fmha_forward_impl(
    sycl::queue& q,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout_mask,
    float dropout_prob,
    T* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys) {
  // RECORD_FUNCTION("fmha_forward_impl", {});
  // fmha forward kernel
  using fmha_forward_op_t =
      fmha_forward_t<fmha_policy, T, kUseBias, kUseDropout>;

  sycl::nd_range<3> NdRange =
      fmha_forward_op_t::get_nd_range(num_batches * num_heads, num_queries);

  //  auto event = q.submit([&](sycl::handler& cgh) {
  //    cgh.parallel_for<
  //        class FmhaForwardKernel<fmha_policy, T, kUseBias, kUseDropout>>(
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NdRange, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      // exec item
      xetla_exec_item<3> ei(item);

      // Init fmha forward op and arguments
      fmha_forward_op_t fmha_fwd_op;
      typename fmha_forward_op_t::arguments_t args(
          query,
          key,
          value,
          bias,
          dropout_mask,
          dropout_prob,
          out,
          num_batches,
          num_heads,
          head_size,
          num_queries,
          num_keys);

      // call the functor
      fmha_fwd_op(ei, &args);
    });
  };
  DPCPP_Q_SUBMIT(q, cgf);
  // event.wait();
  //  double time =
  //      (event.template get_profiling_info<
  //           sycl::info::event_profiling::command_end>() -
  //       event.template get_profiling_info<
  //           sycl::info::event_profiling::command_start>());
  //  uint64_t ops = num_batches * num_heads *
  //      uint64_t(head_size * num_queries * num_keys) * 2L * 2L;
  //  double tflops = (ops / 1024.0f / 1024.0f / 1024.0f / 1024.0f) / (time
  //  / 1e9); printf(
  //      "B, N, F, T, H: %d, %d, %d, %d, %d, time: %f us, tflops: %f\n",
  //      num_batches,
  //      num_heads,
  //      num_queries,
  //      num_keys,
  //      head_size,
  //      time / 1e3,
  //      tflops);
}

} // namespace

#define CALL_IMPL_FUNC(P)                         \
  fmha_forward_impl<P, T, kUseBias, kUseDropout>( \
      q,                                          \
      query,                                      \
      key,                                        \
      value,                                      \
      bias,                                       \
      dropout_mask,                               \
      dropout_prob,                               \
      out,                                        \
      num_batches,                                \
      num_heads,                                  \
      head_size,                                  \
      num_queries,                                \
      num_keys)

/// @brief Main execution function for flash mha forward.
template <typename T, bool kUseBias = false, bool kUseDropout = false>
void fmha_forward(
    sycl::queue& q,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout_mask,
    float dropout_prob,
    T* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys) {
  // RECORD_FUNCTION("fmha_forward", {});
  if (num_queries <= 8 /* inference, common beam */) { // skinny
    if (head_size <= 128) {
      if (num_keys > 64) {
        CALL_IMPL_FUNC(fmha_policy_f1_t1024_h128);
      } else {
        CALL_IMPL_FUNC(fmha_policy_f1_t64_h128);
      }
    } else if (head_size <= 256) {
      if (num_keys > 64) {
        CALL_IMPL_FUNC(fmha_policy_f1_t1024_h256);
      } else {
        CALL_IMPL_FUNC(fmha_policy_f1_t64_h256);
      }
    }
#if 0 // comment for now
  } else if (head_size <= 64) {
    if (num_queries == 512 && num_keys == 512) {
      CALL_IMPL_FUNC(fmha_policy_f512_t512_h64);
    } else if (num_queries == 384 && num_keys == 384) {
      CALL_IMPL_FUNC(fmha_policy_f384_t384_h64);
    } else if (num_queries == 4096 && num_keys == 4096) {
      CALL_IMPL_FUNC(fmha_policy_f4096_t4096_h64);
    } else if (num_queries == 4096 && num_keys == 77) {
      CALL_IMPL_FUNC(fmha_policy_f4096_t77_h64);
    }
  } else if (head_size <= 96) {
    if (num_queries == 1024 && num_keys == 1024) {
      CALL_IMPL_FUNC(fmha_policy_f1024_t1024_h96);
    } else if (num_queries == 1024 && num_keys == 77) {
      CALL_IMPL_FUNC(fmha_policy_f1024_t77_h96);
    }
  } else if (head_size <= 128) {
    CALL_IMPL_FUNC(fmha_policy_h128);
  } else if (head_size <= 160) {
    if (num_queries == 256 && num_keys == 256) {
      CALL_IMPL_FUNC(fmha_policy_f256_t256_h160);
    } else if (num_queries == 256 && num_keys == 77) {
      CALL_IMPL_FUNC(fmha_policy_f256_t77_h160);
    }
    if (num_queries == 64 && num_keys == 64) {
      CALL_IMPL_FUNC(fmha_policy_f64_t64_h160);
    } else if (num_queries == 64 && num_keys == 77) {
      CALL_IMPL_FUNC(fmha_policy_f64_t77_h160);
    }
#endif
  } else if (head_size <= 256) {
    CALL_IMPL_FUNC(fmha_policy_f64_t64_h256);
  } else {
    std::cout << "No policy available for current head_size " << head_size
              << "\n";
    return;
  }
}

#undef CALL_IMPL_FUNC

void fmha_forward_op(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys) {
  // IPEX_DISPATCH_FLOATING_TYPES_AND2(
  //    at::ScalarType::BFloat16,
  //    at::ScalarType::Half,
  //    query.scalar_type(),
  //    "_scaled_dot_product_efficient_attention",
  //    [&] {
  // auto emp_ptr = at::empty_like(query);
  // RECORD_FUNCTION("fmha_forward_op", {});
  using T = sycl::half;
  fmha_forward<T>(
      q,
      (T*)query,
      (T*)key,
      (T*)value,
      (T*)nullptr,
      (uint8_t*)nullptr,
      1.0,
      (T*)out,
      num_batches,
      num_heads,
      head_size,
      num_queries,
      num_keys);
  //  )};
}

} // namespace gpu::xetla
//} // namespace xpu
