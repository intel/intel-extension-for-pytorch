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

#include <experimental/group/gemm/common.hpp>
#include <experimental/group/gemm/compute_policy.hpp>

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief Is the gemm functor for Xe architecture and matrix engine.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename tile_shape_,
    typename mem_desc_a_t_,
    typename mem_desc_b_t_,
    typename dtype_scale_,
    typename dtype_zero_pt_,
    quant_info quant_info_,
    mma_engine mma_engine_,
    typename pre_processing_t_,
    gpu_arch arch_tag_>
class gemm_t<
    compute_policy_int4_dequantize<
        compute_attr_,
        perf_tuning_knob_,
        dtype_scale_,
        dtype_zero_pt_,
        quant_info_,
        mma_engine_,
        arch_tag_>,
    tile_shape_, // tile shape of workgroup-level gemm
    mem_desc_a_t_, // memory attribute of matA
    mem_desc_b_t_, // memory attribute of matB
    pre_processing_t_ // pre_processing functor
    > {
 public:
  using mem_desc_a_t = mem_desc_a_t_;
  using mem_desc_b_t = mem_desc_b_t_;
  using tile_shape = tile_shape_;
  using pre_processing_t = pre_processing_t_;
  using compute_policy = compute_policy_int4_dequantize<
      compute_attr_,
      perf_tuning_knob_,
      dtype_scale_,
      dtype_zero_pt_,
      quant_info_,
      mma_engine_,
      arch_tag_>;
  static constexpr uint32_t k_stride = compute_policy::k_stride;

  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  using work_group_t = typename tile_shape::work_group_t;

  constexpr static gpu_arch arch_tag = compute_policy::arch_tag;
  static constexpr uint32_t dequant_s = compute_policy::dequant_s;
  static constexpr quant_mode quant_mode = compute_policy::quant_mode;
  using dtype_b = typename mem_desc_b_t::dtype;
  using dtype_zero_pt = typename compute_policy::dtype_zero_pt;
  static constexpr uint32_t pack_ratio = sizeof(dtype_b) * 2;

  static constexpr mem_layout mem_layout_a = mem_desc_a_t::layout;
  static constexpr mem_layout mem_layout_b = mem_desc_b_t::layout;
  static constexpr bool is_col_major_a = mem_layout_a == mem_layout::col_major;
  static constexpr bool is_col_major_b = mem_layout_b == mem_layout::col_major;
  static constexpr bool is_gemv = is_col_major_b &&
      compute_policy::mma_engine == mma_engine::fpu && sg_tile_m <= 4;

 private:
  /******** set data type **********/
  using dtype_a = typename mem_desc_a_t::dtype;
  using dtype_mma_acc = typename compute_policy::dtype_mma_acc;
  using dtype_mma_a = typename compute_policy::dtype_mma_a;
  using dtype_mma_b = typename compute_policy::dtype_mma_b;
  using dtype_scale = typename compute_policy::dtype_scale;

  static_assert(
      std::is_same<remove_const_t<dtype_b>, remove_const_t<int4x2>>::value ||
          std::is_same<remove_const_t<dtype_b>, remove_const_t<int4x8>>::value,
      "this is for 4bit matB ");
  static_assert(
      quant_info_.quant_mode == quant_mode::I4_ASYM_FP_ZERO
          ? std::is_same_v<
                remove_const_t<dtype_zero_pt>,
                remove_const_t<dtype_a>>
          : (std::is_same_v<
                 remove_const_t<dtype_zero_pt>,
                 remove_const_t<int4x2>> ||
             std::is_same_v<
                 remove_const_t<dtype_zero_pt>,
                 remove_const_t<int4x8>>),
      "this is for 4bit zero_pt ");

  /******** set memory attribute **********/
  static constexpr mem_space mem_space_a = mem_desc_a_t::space;
  static constexpr mem_space mem_space_b = mem_desc_b_t::space;

  static constexpr bool is_local_a = mem_space_a == mem_space::local;
  static constexpr bool is_local_b = mem_space_b == mem_space::local;
  static constexpr tdesc_update_dir update_dir_a =
      is_col_major_a ? tdesc_update_dir::y_dir : tdesc_update_dir::x_dir;
  static constexpr tdesc_update_dir update_dir_b =
      is_col_major_b ? tdesc_update_dir::x_dir : tdesc_update_dir::y_dir;
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

  static constexpr uint32_t block_size_x_a =
      std::min(compute_policy::block_size_x_a, tile_size_x_a);
  static constexpr uint32_t block_size_y_a =
      std::min(compute_policy::block_size_y_a, tile_size_y_a);
  static constexpr uint32_t block_size_x_b =
      std::min(compute_policy::block_size_x_b, tile_size_x_b);
  static constexpr uint32_t block_size_y_b =
      std::min(compute_policy::block_size_y_b, tile_size_y_b);

  /******** set tile  **********/
  static constexpr bool is_vnni_tiled_a =
      compute_policy::mma_engine == mma_engine::xmx
      ? ((sizeof(dtype_a) < sizeof(uint32_t)) && is_col_major_a)
      : false;

  static constexpr reg_layout reg_layout_a =
      // fpu
      compute_policy::mma_engine == mma_engine::fpu
      ? (is_gemv ? reg_layout::tiled : reg_layout::transpose_tiled)
      // xmx
      : is_vnni_tiled_a ? reg_layout::vnni_tiled
                        : reg_layout::tiled;

  static constexpr reg_layout reg_layout_b =
      // fpu
      compute_policy::mma_engine == mma_engine::fpu
      ? (is_gemv ? reg_layout::transpose_tiled : reg_layout::tiled)
      // xmx
      : (sizeof(dtype_mma_b) < sizeof(uint32_t)) ? reg_layout::vnni_tiled
                                                 : reg_layout::tiled;

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
  using matA_prefetch_payload_t = subgroup::
      prefetch_payload_t<mem_desc_a_t, matA_tile_desc_t, wg_size_x, arch_tag>;

  // note: plane format, row-major
  // note: 4bit x 2, row-major
  using matB_tile_desc_t = std::conditional_t<
      is_col_major_b,
      // compress int4 along K dimensions
      subgroup::tile_desc_t<
          tile_size_x_b,
          tile_size_y_b / pack_ratio,
          block_size_x_b,
          // block_size_y_b * sizeof(dtype_mma_b) / sizeof(dtype_b),
          block_size_y_b / pack_ratio,
          reg_layout_b>,
      // compress int4 along N dimensions
      subgroup::tile_desc_t<
          tile_size_x_b / pack_ratio,
          tile_size_y_b,
          // block_size_x_b * sizeof(dtype_mma_b) / sizeof(dtype_b),
          block_size_x_b / pack_ratio,
          block_size_y_b,
          reg_layout_b>>;
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
      reg_layout_b>;
  using matB_acc_t = subgroup::tile_t<dtype_mma_b, matB_acc_tile_desc_t>;

 public:
  static_assert(
      (k_stride % (block_size_y_b) == 0),
      "k_stride % (block_size_y_b) == 0");
  static_assert(
      (dequant_s % block_size_y_b == 0 || block_size_y_b % dequant_s == 0),
      "dequant_s % block_size_y_b == 0 || block_size_y_b % dequant_s == 0");
  static_assert(
      (k_stride % (dequant_s) == 0) || (dequant_s % (k_stride) == 0),
      "k_stride should match with dequant_s");

  // num_block_y set to 1
  static constexpr uint32_t block_size_y_scale =
      (k_stride + dequant_s - 1) / dequant_s;
  static constexpr uint32_t tile_size_y_scale = block_size_y_scale;
  static constexpr uint32_t block_size_y_zero_pt =
      (k_stride + dequant_s - 1) / dequant_s;
  static constexpr uint32_t tile_size_y_zero_pt = block_size_y_zero_pt;

  static constexpr uint32_t scale_addr_update_freq =
      (k_stride < dequant_s) ? dequant_s / k_stride : 1;

  static constexpr uint32_t zero_pt_addr_update_freq =
      (k_stride < dequant_s) ? dequant_s / k_stride : 1;

  using mem_desc_scale_t = mem_desc_t<
      dtype_scale,
      mem_layout_b,
      mem_space::global,
      mem_desc_b_t::alignment>;

  using mem_desc_zero_pt_t = mem_desc_t<
      dtype_zero_pt,
      mem_layout::row_major,
      mem_space::global,
      mem_desc_b_t::alignment>;

  using matC_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_c,
      tile_size_y_c,
      block_size_x_b,
      block_size_y_a,
      reg_layout::tiled>;
  using matC_t = subgroup::tile_t<dtype_mma_acc, matC_tile_desc_t>;

 private:
  using matAcc_tile_desc_t = subgroup::tile_desc_t<
      block_size_y_b,
      tile_size_y_a,
      block_size_y_b,
      block_size_y_a,
      reg_layout::tiled>;
  using matAcc_t = subgroup::tile_t<dtype_mma_acc, matAcc_tile_desc_t>;
  using scale_tile_desc_t = subgroup::tile_desc_t<
      tile_size_x_b,
      tile_size_y_scale,
      block_size_x_b,
      block_size_y_scale,
      is_col_major_b ? reg_layout::transpose_tiled : reg_layout::tiled>;
  using scale_t = subgroup::tile_t<dtype_scale, scale_tile_desc_t>;
  using scale_payload_t = subgroup::mem_payload_t<
      mem_desc_scale_t,
      scale_tile_desc_t,
      subgroup::msg_type_v<scale_tile_desc_t, mem_desc_scale_t>,
      arch_tag>;

  // compress int4 along N dimensions
  using zero_pt_tile_desc_t = std::conditional_t<
      quant_info_.quant_mode != quant_mode::I4_ASYM_FP_ZERO,
      subgroup::tile_desc_t<
          (tile_size_x_b + pack_ratio - 1) / pack_ratio,
          tile_size_y_zero_pt,
          (block_size_x_b + pack_ratio - 1) / pack_ratio,
          block_size_y_zero_pt,
          reg_layout::tiled>,
      subgroup::tile_desc_t<
          tile_size_x_b,
          tile_size_y_zero_pt,
          block_size_x_b,
          block_size_y_zero_pt,
          reg_layout::tiled>>;

  using zero_pt_t = subgroup::tile_t<dtype_zero_pt, zero_pt_tile_desc_t>;
  using zero_pt_payload_t = subgroup::mem_payload_t<
      mem_desc_zero_pt_t,
      zero_pt_tile_desc_t,
      subgroup::msg_type_v<zero_pt_tile_desc_t, mem_desc_zero_pt_t>,
      arch_tag>;
  using scale_prefetch_payload_t = subgroup::
      prefetch_payload_t<mem_desc_scale_t, scale_tile_desc_t, 1, arch_tag>;
  using zero_pt_prefetch_payload_t = subgroup::
      prefetch_payload_t<mem_desc_zero_pt_t, zero_pt_tile_desc_t, 1, arch_tag>;

  using tile_mma = std::conditional_t<
      is_gemv,
      subgroup::tile_fma_t<matAcc_t, matC_t, matB_acc_t, matA_acc_t, arch_tag>,
      subgroup::tile_mma_t<
          matC_t,
          matC_t,
          matB_acc_t,
          matA_acc_t,
          compute_policy::mma_engine,
          arch_tag>>;
  using dequantize_t = subgroup::dequant_int4_weight_t<
      matB_acc_t,
      matB_t,
      scale_t,
      zero_pt_t,
      dequant_s,
      quant_mode>;
  static constexpr bool enable_periodic_sync = (sync_freq != 0);
  static constexpr uint32_t barrier_count_x = wg_size_y > 1 ? wg_size_x : 0;
  static constexpr uint32_t barrier_count_y = wg_size_x > 1 ? wg_size_y : 0;
  uint32_t wg_start_m = 0;
  uint32_t wg_start_n = 0;
  uint32_t wg_start_k = 0;

 public:
  static constexpr uint32_t barrier_count =
      enable_periodic_sync && arch_has_named_barrier<arch_tag>
      ? barrier_count_x + barrier_count_y
      : 0;
  // current only support matA from slm
  static constexpr uint32_t slm_size =
      is_local_a ? sg_tile_m * wg_size_y * k_stride * sizeof(dtype_a) : 0;

  static constexpr msg_type msg_type_a = matA_payload_t::message_type;
  static constexpr msg_type msg_type_b = matB_payload_t::message_type;

  /// @brief Arguments for gemm.
  /// User should prepare matA_base_desc, matB_base_desc,
  /// inner_loop_start inner_loop_count...
  struct arguments_t {
    /// @brief Is the memory description of matA, including base, shape and
    /// coordinate.
    mem_desc_a_t matA_base_desc;
    /// @brief Is the memory description of matB, including base, shape and
    /// coordinate.
    mem_desc_b_t matB_base_desc;
    /// @brief The tile starting from K-dim
    uint32_t inner_loop_start;
    /// @brief Is the total inner loop count required to compute the entire
    /// K-dim.
    uint32_t inner_loop_count;
    /// @brief Is the memory description of scale buffer. Scale size:
    /// (matrix_k/dequant_s)x(matrix_n)
    mem_desc_scale_t scale_base_desc;
    /// @brief Is the memory description of zero_pt buffer. Zero_pt size:
    /// (matrix_k/dequant_s)x(matrix_n/pack_ratio)
    mem_desc_zero_pt_t zero_pt_base_desc;

    /// @brief Default construct.
    inline arguments_t() = default;

    inline arguments_t(
        mem_desc_a_t matA_desc,
        mem_desc_b_t matB_desc,
        uint32_t loop_start,
        uint32_t loop_count,
        mem_desc_scale_t scale_desc,
        mem_desc_zero_pt_t zero_pt_desc)
        : matA_base_desc(matA_desc),
          matB_base_desc(matB_desc),
          inner_loop_start(loop_start),
          inner_loop_count(loop_count),
          scale_base_desc(scale_desc),
          zero_pt_base_desc(zero_pt_desc) {}

    inline arguments_t(
        mem_desc_a_t matA_desc,
        mem_desc_b_t matB_desc,
        uint32_t loop_start,
        uint32_t loop_count,
        mem_desc_scale_t scale_desc)
        : matA_base_desc(matA_desc),
          matB_base_desc(matB_desc),
          inner_loop_start(loop_start),
          inner_loop_count(loop_count),
          scale_base_desc(scale_desc) {}
    // Be aware of the risks: Rule of three (copy constructor, copy
    // assignment, destructor) Please check if you need to add self-define
    // destructor inline ~arguments_t(){}
    inline arguments_t(const arguments_t& args)
        : matA_base_desc(args.matA_base_desc),
          matB_base_desc(args.matB_base_desc),
          inner_loop_start(args.inner_loop_start),
          inner_loop_count(args.inner_loop_count),
          scale_base_desc(args.scale_base_desc),
          zero_pt_base_desc(args.zero_pt_base_desc) {}
    inline arguments_t& operator=(const arguments_t& args) {
      this->matA_base_desc = args.matA_base_desc;
      this->matB_base_desc = args.matB_base_desc;
      this->inner_loop_start = args.inner_loop_start;
      this->inner_loop_count = args.inner_loop_count;
      this->scale_base_desc = args.scale_base_desc;
      this->zero_pt_base_desc = args.zero_pt_base_desc;
      return *this;
    }
    inline void init(
        mem_desc_a_t matA_desc,
        mem_desc_b_t matB_desc,
        uint32_t loop_start,
        uint32_t loop_count,
        mem_desc_scale_t scale_desc,
        mem_desc_zero_pt_t zero_pt_desc) {
      matA_base_desc = matA_desc;
      matB_base_desc = matB_desc;
      inner_loop_start = loop_start;
      inner_loop_count = loop_count;
      scale_base_desc = scale_desc;
      zero_pt_base_desc = zero_pt_desc;
    }
  };

  /// @brief Gets the subgroup-level tile offset x.
  /// @param g Is the workgroup of the current tile.
  /// @return Subgroup-level tile offset x.
  __XETLA_API static int get_matC_offset_x(work_group_t& g) {
    int32_t sg_idx = g.get_id() % wg_size_x;
    return sg_idx * sg_tile_n;
  }

  /// @brief Gets the subgroup-level tile offset y.
  /// @param g Is the workgroup of the current tile.
  /// @return Subgroup-level tile offset y.
  __XETLA_API static int get_matC_offset_y(work_group_t& g) {
    int32_t sg_idy = g.get_id() / wg_size_x;
    return sg_idy * sg_tile_m;
  }

  XETLA_MARKER(
      "This release function will wait until all the  r/w and nbarrier "
      "id used in this gemm have been committed. By default, it will "
      "use barrier_id 0 to do the entire workgroup sync if wg_size > 1. "
      "If you call this function, please set a free barrier id or make "
      "sure barrier_id 0 is not being occupied and you need to allocate "
      "one more barrier count in addition to the gemm barrier counts.")
  __XETLA_API static void release(uint8_t nbarrier_id = 0) {
    static constexpr bool need_local_fence =
        (mem_space_a == mem_space::local) || (mem_space_b == mem_space::local);
    if constexpr (need_local_fence) {
      xetla_fence<memory_kind::shared_local>();
    }
    xetla_fence<memory_kind::untyped_global>();
    static constexpr uint32_t wg_size = wg_size_x * wg_size_y;
    if constexpr (wg_size > 1) {
      xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;
      nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
      nbarrier.arrive_wait();
    }
  }

  /// @brief Main execution function for gemm.
  /// The basic process is load data -> matrix multiply.
  /// @param g Is the workgroup of the current tile.
  /// @param matC Is the reference of the accumulation buffer.
  /// @param args Is the gemm::arguments_t.
  /// @param slm_base Is the slm base address.
  /// @param nbarrier_base Is the named barrier base.
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g,
      matC_t& matC,
      arguments_t args,
      [[maybe_unused]] uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    int32_t sg_idx = g.get_id() % wg_size_x;
    int32_t sg_idy = g.get_id() / wg_size_x;
    update_sg_tile_tdesc(args, sg_idx, sg_idy);

    matA_t matA;
    matB_t matB;
    scale_t scale;
    zero_pt_t zero_pt;
    matAcc_t matAcc;
    matAcc.reg = 0;

    matA_payload_t matA_payload(args.matA_base_desc);
    matB_payload_t matB_payload(args.matB_base_desc);
    scale_payload_t scale_payload(args.scale_base_desc);
    zero_pt_payload_t zero_pt_payload(args.zero_pt_base_desc);
    matA_prefetch_payload_t matA_prefetch_payload(args.matA_base_desc, sg_idx);
    matB_prefetch_payload_t matB_prefetch_payload(args.matB_base_desc, sg_idy);
    scale_prefetch_payload_t scale_prefetch_payload(args.scale_base_desc, 0);
    zero_pt_prefetch_payload_t zero_pt_prefetch_payload(
        args.zero_pt_base_desc, 0);

    wg_start_m = args.matA_base_desc.coord.y;
    wg_start_n = args.scale_base_desc.coord.x;
    wg_start_k = args.matA_base_desc.coord.x;
    typename dequantize_t::arguments_t dequantize_args{wg_start_n, wg_start_k};
    dequantize_t dequantize;

    xetla_nbarrier_t<wg_size_x, wg_size_x, arch_tag> nbarrier_a;
    nbarrier_a.init_nbarrier(
        sg_idy + nbarrier_base, nbarrier_role::producer_consumer);
    xetla_nbarrier_t<wg_size_y, wg_size_y, arch_tag> nbarrier_b;
    nbarrier_b.init_nbarrier(
        sg_idx + barrier_count_y + nbarrier_base,
        nbarrier_role::producer_consumer);

    int scale_prefetch_addr_i = args.inner_loop_start;
    int tile_k_idx = args.inner_loop_start;
    uint32_t prefetch_stages =
        stages < args.inner_loop_count ? stages : args.inner_loop_count;
    uint32_t prefetch_compute_stages =
        stages < args.inner_loop_count ? args.inner_loop_count - stages : 0;
    uint32_t compute_stages =
        stages < args.inner_loop_count ? stages : args.inner_loop_count;
    for (uint32_t i = 0; i < prefetch_stages; i++) {
      subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          matA_prefetch_payload);
      subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          matB_prefetch_payload);
      // TODO 1D prefetch need pack to U32/U64
      subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
          scale_prefetch_payload);
      if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
        // TODO 1D prefetch need pack to U32/U64
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            zero_pt_prefetch_payload);
      }
      scale_prefetch_addr_i++;
      matA_prefetch_payload.template update_tdesc<update_dir_a>(
          matA_t::tile_size_x);
      matB_prefetch_payload.template update_tdesc<update_dir_b>(
          matB_t::tile_size_y);
      if ((scale_prefetch_addr_i % scale_addr_update_freq) == 0) {
        scale_prefetch_payload.template update_tdesc<update_dir_b>(
            scale_t::tile_size_y);
        if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
          zero_pt_prefetch_payload
              .template update_tdesc<tdesc_update_dir::y_dir>(
                  zero_pt_t::tile_size_y);
        }
      }
    }
    for (uint32_t i = 0; i < prefetch_compute_stages; i++) {
      if constexpr (enable_periodic_sync) {
        if ((i % sync_freq) == 0) {
          if constexpr (wg_size_x > 1) {
            nbarrier_a.arrive();
          }
          if constexpr (arch_has_named_barrier<arch_tag>) {
            if constexpr (wg_size_y > 1) {
              nbarrier_b.arrive();
            }
          }
        }
      }
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matA, matA_payload);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matB, matB_payload);
      // subgroup::tile_load<cache_hint::uncached, cache_hint::uncached>(
      //     matB, matB_payload);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          scale, scale_payload);
      if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
            zero_pt, zero_pt_payload);
      }
      tile_k_idx++;
      if constexpr (stages != 0) {
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            matA_prefetch_payload);
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            matB_prefetch_payload);
        // TODO 1D prefetch need pack to U32/U64
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            scale_prefetch_payload);
        if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
          // TODO 1D prefetch need pack to U32/U64
          subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
              zero_pt_prefetch_payload);
        }
        scale_prefetch_addr_i++;
      }
      matA_payload.template update_tdesc<update_dir_a>(matA_t::tile_size_x);
      matB_payload.template update_tdesc<update_dir_b>(matB_t::tile_size_y);
      if (tile_k_idx % scale_addr_update_freq == 0) {
        scale_payload.template update_tdesc<update_dir_b>(scale_t::tile_size_y);
      }
      if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
        if (tile_k_idx % zero_pt_addr_update_freq == 0) {
          zero_pt_payload.template update_tdesc<tdesc_update_dir::y_dir>(
              zero_pt_t::tile_size_y);
        }
      }
      if constexpr (stages != 0) {
        matA_prefetch_payload.template update_tdesc<update_dir_a>(
            matA_t::tile_size_x);
        matB_prefetch_payload.template update_tdesc<update_dir_b>(
            matB_t::tile_size_y);
        if ((scale_prefetch_addr_i % scale_addr_update_freq) == 0) {
          scale_prefetch_payload.template update_tdesc<tdesc_update_dir::y_dir>(
              scale_t::tile_size_y);
          if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
            zero_pt_prefetch_payload
                .template update_tdesc<tdesc_update_dir::y_dir>(
                    zero_pt_t::tile_size_y);
          }
        }
      }
      matA_acc_t matA_acc;
      matB_acc_t matB_acc;
      if constexpr (is_vnni_tiled_a) {
        subgroup::vnni_reverse(matA);
      }
      subgroup::elemwise_cvt(matA_acc, matA);

      dequantize(matB_acc, matB, scale, zero_pt, dequantize_args);
      if constexpr (is_gemv) {
        tile_mma::mma(
            matAcc,
            matAcc,
            matC,
            matB_acc,
            matA_acc,
            i == args.inner_loop_count - 1);
      } else {
        if constexpr (is_col_major_b) {
          tile_transpose(matB_acc);
        }
        tile_mma::mma(matC, matC, matB_acc, matA_acc);
      }
      if constexpr (enable_periodic_sync) {
        if ((i % sync_freq) == 0) {
          if constexpr (wg_size_x > 1) {
            nbarrier_a.wait();
          }
          if constexpr (arch_has_named_barrier<arch_tag>) {
            if constexpr (wg_size_y > 1) {
              nbarrier_b.wait();
            }
          }
        }
      }
    }
    for (uint32_t i = 0; i < compute_stages; i++) {
      if constexpr (enable_periodic_sync) {
        if ((i % sync_freq) == 0) {
          if constexpr (wg_size_x > 1) {
            nbarrier_a.arrive();
          }
          if constexpr (arch_tag >= gpu_arch::XeHpc) {
            if constexpr (wg_size_y > 1) {
              nbarrier_b.arrive();
            }
          }
        }
      }
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matA, matA_payload);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          matB, matB_payload);
      // subgroup::tile_load<cache_hint::uncached, cache_hint::uncached>(
      //     matB, matB_payload);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          scale, scale_payload);
      if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
            zero_pt, zero_pt_payload);
      }
      tile_k_idx++;
      matA_payload.template update_tdesc<update_dir_a>(matA_t::tile_size_x);
      matB_payload.template update_tdesc<update_dir_b>(matB_t::tile_size_y);
      if (tile_k_idx % scale_addr_update_freq == 0) {
        scale_payload.template update_tdesc<update_dir_b>(scale_t::tile_size_y);
      }
      if constexpr (compute_policy::quant_mode != quant_mode::I4_SYM) {
        if (tile_k_idx % zero_pt_addr_update_freq == 0) {
          zero_pt_payload.template update_tdesc<tdesc_update_dir::y_dir>(
              zero_pt_t::tile_size_y);
        }
      }
      matA_acc_t matA_acc;
      matB_acc_t matB_acc;
      if constexpr (is_vnni_tiled_a) {
        subgroup::vnni_reverse(matA);
      }
      subgroup::elemwise_cvt(matA_acc, matA);

      dequantize(matB_acc, matB, scale, zero_pt, dequantize_args);
      if constexpr (is_gemv) {
        tile_mma::mma(
            matAcc, matAcc, matC, matB_acc, matA_acc, i == compute_stages - 1);
      } else {
        if constexpr (is_col_major_b) {
          tile_transpose(matB_acc);
        }
        tile_mma::mma(matC, matC, matB_acc, matA_acc);
      }
      if constexpr (enable_periodic_sync) {
        if ((i % sync_freq) == 0) {
          if constexpr (wg_size_x > 1) {
            nbarrier_a.wait();
          }
          if constexpr (arch_tag >= gpu_arch::XeHpc) {
            if constexpr (wg_size_y > 1) {
              nbarrier_b.wait();
            }
          }
        }
      }
    }
  }

 private:
  /// @brief Updates tile base descriptor based on the tid.
  __XETLA_API static void update_sg_tile_tdesc(
      arguments_t& args,
      int32_t sg_idx,
      int32_t sg_idy) {
    int32_t tile_offset_n = sg_idx * sg_tile_n;
    int32_t tile_offset_m = sg_idy * sg_tile_m;

    args.matA_base_desc.update_coord_y(tile_offset_m);
    if constexpr (is_col_major_b) {
      args.matB_base_desc.update_coord_x(tile_offset_n);
    } else {
      args.matB_base_desc.update_coord_x(tile_offset_n / pack_ratio);
    }
    args.scale_base_desc.update_coord_x(tile_offset_n);
    args.zero_pt_base_desc.update_coord_x(tile_offset_n / pack_ratio);
  }
};
/// @} xetla_gemm

} // namespace gpu::xetla::group
