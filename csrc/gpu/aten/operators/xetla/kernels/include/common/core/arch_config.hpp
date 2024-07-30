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

#include <common/core/common.hpp>

namespace gpu::xetla {

/// @addtogroup xetla_core_arch_config
/// @{

template <msg_type message_type, gpu_arch arch_tag>
struct load_store_attr_t {
  static constexpr bool has_hw_block_2d = false;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpc> {
  /// HW limitation checks https://gfxspecs.intel.com/Predator/Home/Index/55490
  static constexpr bool has_hw_block_2d = true;
  static constexpr uint32_t max_load_height_in_elem = 32;
  static constexpr uint32_t max_load_width_in_bytes = 64;
  static constexpr uint32_t max_trans_load_width_in_bytes = 32;
  static constexpr uint32_t max_vnni_load_width_in_elems = 16;
  static constexpr uint32_t min_vnni_load_height_in_bytes = 4;

  static constexpr uint32_t max_store_height_in_elem = 8;
  static constexpr uint32_t max_store_width_in_bytes = 64;

  static constexpr uint32_t max_load_size_in_bytes = 2048;
  static constexpr uint32_t max_store_size_in_bytes = 512;

  static constexpr uint32_t special_prefetch_width_in_bytes = 64;

  static constexpr uint32_t cache_line_size_in_bytes = 64;
  static constexpr uint32_t alignment_in_bytes = 16;
};

template <msg_type message_type, gpu_arch arg_tag>
struct client_load_store_attr_base_t {
  /// HW limitation checks https://gfxspecs.intel.com/Predator/Home/Index/55490
  static constexpr bool has_hw_block_2d = false;
  static constexpr uint32_t max_load_height_in_elem = 32;
  static constexpr uint32_t max_load_width_in_bytes = 64;
  static constexpr uint32_t max_trans_load_width_in_bytes = 32;
  static constexpr uint32_t max_vnni_load_width_in_elems = 16;
  static constexpr uint32_t min_vnni_load_height_in_bytes = 4;

  static constexpr uint32_t max_store_height_in_elem = 8;
  static constexpr uint32_t max_store_width_in_bytes = 64;

  static constexpr uint32_t max_load_size_in_bytes = 2048;
  static constexpr uint32_t max_store_size_in_bytes = 512;

  static constexpr uint32_t special_prefetch_width_in_bytes = 64;

  static constexpr uint32_t cache_line_size_in_bytes = 64;
  static constexpr uint32_t alignment_in_bytes = 4;
};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeHpg>
    : public client_load_store_attr_base_t<
          msg_type::block_2d,
          gpu_arch::XeHpg> {};

template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::XeLpg>
    : public client_load_store_attr_base_t<
          msg_type::block_2d,
          gpu_arch::XeLpg> {};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_2d_load_store =
    load_store_attr_t<msg_type::block_2d, arch_tag>::has_hw_block_2d;

template <gpu_arch arch_tag>
struct load_store_attr_t<msg_type::block_1d, arch_tag> {
  static constexpr uint32_t max_load_vec_len = 256;
  static constexpr uint32_t max_aligned_load_vec_len = 256;
  static constexpr uint32_t max_store_vec_len = 256;
  static constexpr uint32_t max_aligned_store_vec_len = 256;
  static constexpr uint32_t max_prefetch_vec_len = 32;
  static constexpr uint32_t max_channel_num = 16;
};

template <>
struct load_store_attr_t<msg_type::block_1d, gpu_arch::XeHpc> {
  static constexpr uint32_t max_load_vec_len = 256;
  static constexpr uint32_t max_aligned_load_vec_len = 512;
  static constexpr uint32_t max_store_vec_len = 256;
  static constexpr uint32_t max_aligned_store_vec_len = 512;
  static constexpr uint32_t max_prefetch_vec_len = 64;
  static constexpr uint32_t max_channel_num = 32;
};

struct dpas_attr_base_t {
  static constexpr bool has_xmx = true;
  static constexpr uint32_t systolic_depth = 8;
  static constexpr uint32_t rcount_max = 8;
  static constexpr uint32_t op_per_channel_bits = 32;
  static constexpr uint32_t op_per_channel_bytes = (op_per_channel_bits >> 3);
  static constexpr uint32_t op_per_channel_max = 8;
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
struct dpas_attr_t<gpu_arch::XeHpg> : public dpas_attr_base_t {
  static constexpr uint32_t n_in_elem = 8;
};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_xmx = dpas_attr_t<arch_tag>::has_xmx;

template <gpu_arch arch_tag>
struct fpu_attr_t {
  static constexpr bool has_fpu = true;
};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_fpu = fpu_attr_t<arch_tag>::has_fpu;

#ifdef NORMAL_GRF
#define GRF grf_mode::normal_grf
#else
#define GRF grf_mode::double_grf
#endif

template <grf_mode grf_num_mode>
struct register_nums_t {
  static constexpr uint32_t register_nums =
      (grf_num_mode == grf_mode::normal) ? 128 : 256;
  static constexpr uint32_t acc_register_nums =
      (grf_num_mode == grf_mode::normal) ? 4 : 8;
};

template <gpu_arch arch_tag>
struct register_bytes_t;

template <>
struct register_bytes_t<gpu_arch::XeHpc> {
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
      load_store_attr::max_trans_load_width_in_bytes;

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

template <gpu_arch arch_tag>
struct arch_attr_t {};

template <>
struct arch_attr_t<gpu_arch::XeHpc> {
  template <msg_type message_type = msg_type::block_2d>
  using load_store_attr = load_store_attr_t<message_type, gpu_arch::XeHpc>;

  template <grf_mode grf_num_mode = GRF>
  using register_attr = register_attr_t<grf_num_mode, gpu_arch::XeHpc>;

  using dpas_attr = dpas_attr_t<gpu_arch::XeHpc>;

  static constexpr uint32_t max_wg_num = 64;
  static constexpr uint32_t local_mem_size = 128 * 1024;
  static constexpr bool has_named_barrier = true;
};

template <>
struct arch_attr_t<gpu_arch::XeHpg> {
  template <msg_type message_type = msg_type::block_2d>
  using load_store_attr = load_store_attr_t<message_type, gpu_arch::XeHpg>;

  template <grf_mode grf_num_mode = GRF>
  using register_attr = register_attr_t<grf_num_mode, gpu_arch::XeHpg>;

  using dpas_attr = dpas_attr_t<gpu_arch::XeHpg>;

  static constexpr uint32_t max_wg_num = 32;
  static constexpr uint32_t local_mem_size = 64 * 1024;

  static constexpr bool has_named_barrier = false;
};

template <>
struct arch_attr_t<gpu_arch::XeLpg> {
  template <msg_type message_type = msg_type::block_2d>
  using load_store_attr = load_store_attr_t<message_type, gpu_arch::XeLpg>;

  template <grf_mode grf_num_mode = GRF>
  using register_attr = register_attr_t<grf_num_mode, gpu_arch::XeLpg>;

  using dpas_attr = dpas_attr_t<gpu_arch::XeLpg>;

  static constexpr uint32_t max_wg_num = 32;
  static constexpr uint32_t local_mem_size = 64 * 1024;
  static constexpr bool has_named_barrier = false;
};

template <gpu_arch arch_tag>
inline constexpr bool arch_has_named_barrier =
    arch_attr_t<arch_tag>::has_named_barrier;

/// @} xetla_core_arch_config

} // namespace gpu::xetla
