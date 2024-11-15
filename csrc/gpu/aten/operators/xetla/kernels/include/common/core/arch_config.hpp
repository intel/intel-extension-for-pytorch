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

#include "./common.hpp"

namespace gpu::xetla {

/// @addtogroup xetla_core_arch_config
/// @{

/// Load and Store attr for block_2d
template <msg_type message_type, gpu_arch arch_tag>
struct load_store_attr_t {
  static constexpr bool has_hw_block_2d = false;
};

struct load_store_has_block2d_attr_base_t {
  /// HW limitation checks https://gfxspecs.intel.com/Predator/Home/Index/55490
  static constexpr bool has_hw_block_2d = true;
  static constexpr uint32_t max_load_height_in_elem = 32;
  static constexpr uint32_t max_load_width_in_bytes = 64;
  static constexpr uint32_t max_trans_load_width_in_bytes = 32;
  static constexpr uint32_t max_vnni_load_width_in_elems = 16;
  static constexpr uint32_t min_vnni_load_height_in_bytes = 4;
  static constexpr uint32_t max_store_height_in_elem = 8;
  static constexpr uint32_t max_store_width_in_bytes = 64;
  static constexpr uint32_t special_prefetch_width_in_bytes = 64;
  static constexpr uint32_t cache_line_size_in_bytes = 64;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpc>
    : public load_store_has_block2d_attr_base_t {
  static constexpr uint32_t alignment_in_bytes = 8;
  static constexpr uint32_t base_align_in_bytes = 64;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpc_vg>
    : public load_store_has_block2d_attr_base_t {
  static constexpr uint32_t alignment_in_bytes = 8;
  static constexpr uint32_t base_align_in_bytes = 64;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::Xe2Hpg>
    : public load_store_has_block2d_attr_base_t {
  static constexpr uint32_t alignment_in_bytes = 4;
  static constexpr uint32_t base_align_in_bytes = 4;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::Xe2Lpg>
    : public load_store_has_block2d_attr_base_t {
  static constexpr uint32_t alignment_in_bytes = 4;
  static constexpr uint32_t base_align_in_bytes = 4;
};

struct load_store_no_block2d_attr_base_t {
  /// HW limitation checks
  /// https://gfxspecs.intel.com/Predator/Home/Index/55490
  static constexpr bool has_hw_block_2d = false;
  static constexpr uint32_t max_load_height_in_elem = 32;
  static constexpr uint32_t max_load_width_in_bytes = 64;
  static constexpr uint32_t max_trans_load_width_in_bytes = 32;
  static constexpr uint32_t max_vnni_load_width_in_elems = 16;
  static constexpr uint32_t min_vnni_load_height_in_bytes = 4;
  static constexpr uint32_t max_store_height_in_elem = 8;
  static constexpr uint32_t max_store_width_in_bytes = 64;
  static constexpr uint32_t special_prefetch_width_in_bytes = 64;
  static constexpr uint32_t cache_line_size_in_bytes = 64;
  static constexpr uint32_t alignment_in_bytes = 4;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpg>
    : public load_store_no_block2d_attr_base_t {};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeLpg>
    : public load_store_no_block2d_attr_base_t {};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_2d_load_store =
    load_store_attr_t<msg_type::block_2d, arch_tag>::has_hw_block_2d;

/// Load and Store attr for block_1d
template <gpu_arch arch_tag>
struct load_store_attr_t<msg_type::block_1d, arch_tag> {
  static constexpr uint32_t max_load_vec_len = 256;
  static constexpr uint32_t max_store_vec_len = 256;
  static constexpr uint32_t max_aligned_load_vec_len = 256;
  static constexpr uint32_t max_aligned_store_vec_len = 256;
  static constexpr uint32_t max_prefetch_vec_len = 32;
  static constexpr uint32_t max_channel_num = 16;
};

template <>
struct load_store_attr_t<msg_type::block_1d, gpu_arch::XeHpc> {
  static constexpr uint32_t max_load_vec_len = 256;
  static constexpr uint32_t max_store_vec_len = 256;
  static constexpr uint32_t max_aligned_load_vec_len = 512;
  static constexpr uint32_t max_aligned_store_vec_len = 512;
  static constexpr uint32_t max_prefetch_vec_len = 64;
  static constexpr uint32_t max_channel_num = 32;
};

/// dpas attr
struct dpas_attr_base_t {
  static constexpr bool has_xmx = true;
  static constexpr uint32_t rcount_max = 8;
  static constexpr uint32_t op_per_channel_bits = 32;
  static constexpr uint32_t op_per_channel_bytes = (op_per_channel_bits >> 3);
  static constexpr uint32_t op_per_channel_max = 8;
  static constexpr uint32_t systolic_depth = 8;
  static constexpr uint32_t k_in_bytes = systolic_depth * op_per_channel_bytes;
};

template <gpu_arch arch_tag>
struct dpas_attr_t {
  static constexpr bool has_xmx = false;
};

template <>
struct dpas_attr_t<gpu_arch::XeHpc> : public dpas_attr_base_t {
  static constexpr uint32_t n_in_elem = 16;
};

template <>
struct dpas_attr_t<gpu_arch::Xe2Hpg> : public dpas_attr_base_t {
  static constexpr uint32_t n_in_elem = 16;
};

template <>
struct dpas_attr_t<gpu_arch::Xe2Lpg> : public dpas_attr_base_t {
  static constexpr uint32_t n_in_elem = 16;
};

template <>
struct dpas_attr_t<gpu_arch::XeHpg> : public dpas_attr_base_t {
  static constexpr uint32_t n_in_elem = 8;
};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_xmx = dpas_attr_t<arch_tag>::has_xmx;

/// fpu attr
template <gpu_arch arch_tag>
struct fpu_attr_t {
  static constexpr bool has_fpu = true;
};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_fpu = fpu_attr_t<arch_tag>::has_fpu;

#ifdef USE_DOUBLE_GRF
#define GRF grf_mode::double_grf
#else
#define GRF grf_mode::normal_grf
#endif

/// register attr
template <grf_mode grf_num_mode>
struct register_nums_t {
  static constexpr uint32_t register_nums =
      (grf_num_mode == grf_mode::normal_grf) ? 128 : 256;
  static constexpr uint32_t acc_register_nums =
      (grf_num_mode == grf_mode::normal_grf) ? 4 : 8;
};

template <gpu_arch arch_tag>
struct register_bytes_t {
  static constexpr uint32_t reg_in_bytes = 64;
};

template <>
struct register_bytes_t<gpu_arch::XeHpg> {
  static constexpr uint32_t reg_in_bytes = 32;
};
template <>
struct register_bytes_t<gpu_arch::XeLpg> {
  static constexpr uint32_t reg_in_bytes = 32;
};

template <grf_mode grf_num_mode, gpu_arch arch_tag>
struct register_attr_t {
  static constexpr uint32_t reg_in_bytes =
      register_bytes_t<arch_tag>::reg_in_bytes;
  static constexpr uint32_t register_nums =
      register_nums_t<grf_num_mode>::register_nums;
  static constexpr uint32_t acc_register_nums =
      register_nums_t<grf_num_mode>::acc_register_nums;
  static constexpr uint32_t acc_reg_in_bytes = acc_register_nums * reg_in_bytes;

  static constexpr uint32_t grf_in_bytes = register_nums * reg_in_bytes;
};

/// mma attr
template <
    gpu_arch arch_tag,
    mma_engine engine_type,
    uint32_t m,
    class enable = void>
struct mma_attr_t {};

template <gpu_arch arch_tag, uint32_t m>
struct mma_attr_t<
    arch_tag,
    mma_engine::xmx,
    m,
    std::enable_if_t<arch_has_xmx<arch_tag>>> {
  using dpas_attr = dpas_attr_t<arch_tag>;
  using load_store_attr = load_store_attr_t<msg_type::block_2d, arch_tag>;
  static constexpr uint32_t mma_m_in_elem =
      (m > dpas_attr::rcount_max) ? dpas_attr::rcount_max : m;
  static constexpr uint32_t blk_m_in_elem = 16;

  static constexpr uint32_t mma_n_in_elem = dpas_attr::n_in_elem;
  [[maybe_unused]] static constexpr uint32_t blk_n_in_bytes =
      load_store_attr::max_vnni_load_width_in_elems;

  static constexpr uint32_t mma_k_in_bytes = dpas_attr::k_in_bytes;
  static constexpr uint32_t blk_k_in_bytes = mma_k_in_bytes;
};

template <gpu_arch arch_tag, uint32_t m>
struct mma_attr_t<
    arch_tag,
    mma_engine::fpu,
    m,
    std::enable_if_t<arch_has_fpu<arch_tag>>> {
  using load_store_attr = load_store_attr_t<msg_type::block_2d, arch_tag>;
  static constexpr uint32_t mma_m_in_elem = (m > 8) ? 8 : m;
  static constexpr uint32_t blk_m_in_elem = 16;

  static constexpr uint32_t mma_k_in_bytes = 32;
  static constexpr uint32_t blk_k_in_bytes = mma_k_in_bytes;

  [[maybe_unused]] static constexpr uint32_t mma_n_in_elem = 16;
  static constexpr uint32_t blk_n_in_bytes =
      register_bytes_t<arch_tag>::reg_in_bytes;
};

/// named barrier attr
template <gpu_arch arch_tag>
struct named_barrier_attr_t {
  static constexpr bool has_named_barrier = true;
};

template <>
struct named_barrier_attr_t<gpu_arch::XeHpg> {
  static constexpr bool has_named_barrier = false;
};

template <>
struct named_barrier_attr_t<gpu_arch::XeLpg> {
  static constexpr bool has_named_barrier = false;
};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_named_barrier =
    named_barrier_attr_t<arch_tag>::has_named_barrier;

/// local memory attr
template <gpu_arch arch_tag>
struct lm_attr_t {
  static constexpr uint32_t local_mem_size = 128 * 1024;
};

template <>
struct lm_attr_t<gpu_arch::XeLpg> {
  static constexpr uint32_t local_mem_size = 64 * 1024;
};

/// work group attr
template <gpu_arch arch_tag>
struct wg_attr_t {};

template <>
struct wg_attr_t<gpu_arch::XeLpg> {
  static constexpr uint32_t max_wg_num = 8;
  static constexpr uint32_t core_per_ss = 16;
};

template <>
struct wg_attr_t<gpu_arch::XeHpg> {
  static constexpr uint32_t max_wg_num = 32;
  static constexpr uint32_t core_per_ss = 16;
};

template <>
struct wg_attr_t<gpu_arch::XeHpc> {
  static constexpr uint32_t max_wg_num = 64;
  static constexpr uint32_t core_per_ss = 8;
};

template <>
struct wg_attr_t<gpu_arch::Xe2Lpg> {
  static constexpr uint32_t max_wg_num = 8;
  static constexpr uint32_t core_per_ss = 8;
};

template <>
struct wg_attr_t<gpu_arch::Xe2Hpg> {
  static constexpr uint32_t max_wg_num = 40;
  static constexpr uint32_t core_per_ss = 8;
};

/// arch attr
template <gpu_arch arch_tag>
struct arch_attr_t {
  template <msg_type message_type = msg_type::block_2d>
  using load_store_attr = load_store_attr_t<message_type, arch_tag>;

  template <grf_mode grf_num_mode = GRF>
  using register_attr = register_attr_t<grf_num_mode, arch_tag>;

  using dpas_attr = dpas_attr_t<arch_tag>;

  static constexpr uint32_t local_mem_size =
      lm_attr_t<arch_tag>::local_mem_size;

  static constexpr uint32_t max_wg_num = wg_attr_t<arch_tag>::max_wg_num;
  static constexpr uint32_t thread_per_wg = (GRF == grf_mode::normal_grf)
      ? wg_attr_t<arch_tag>::core_per_ss * 8
      : wg_attr_t<arch_tag>::core_per_ss * 4;
};

/// @} xetla_core_arch_config

} // namespace gpu::xetla
