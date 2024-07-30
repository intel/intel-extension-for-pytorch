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

namespace gpu::xetla::group {

template <
    typename T,
    uint32_t SZ,
    uint32_t N,
    reduce_op Op,
    uint32_t N_SG,
    bool is_all_reduce,
    gpu_arch arch_tag_>
struct group_reduce_t {
  static constexpr gpu_arch arch_tag = arch_tag_;
  group_reduce_t<T, SZ, N, Op, 1, is_all_reduce, arch_tag> sg_reduce{};
  xetla_nbarrier_t<N_SG, N_SG, arch_tag> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  using local_st_tile_desc =
      subgroup::tile_desc_t<N, 1, N, 1, reg_layout::tiled>;
  using local_ld_tile_desc =
      subgroup::tile_desc_t<N_SG * N, 1, N_SG * N, 1, reg_layout::tiled>;
  using local_ld_t = subgroup::tile_t<T, local_ld_tile_desc>;
  using local_st_t = subgroup::tile_t<T, local_st_tile_desc>;
  using mem_desc_ld_t = mem_desc_t<T, mem_layout::row_major, mem_space::local>;
  using local_ld_payload_t = subgroup::mem_payload_t<
      mem_desc_ld_t,
      local_ld_tile_desc,
      subgroup::msg_type_v<local_ld_tile_desc, mem_desc_ld_t>,
      arch_tag>;
  using mem_desc_st_t = mem_desc_t<T, mem_layout::row_major, mem_space::local>;
  using local_st_payload_t = subgroup::mem_payload_t<
      mem_desc_st_t,
      local_st_tile_desc,
      // subgroup::msg_type_v<local_ld_tile_desc, mem_desc_st_t>,
      msg_type::block_1d,
      arch_tag>;
  inline group_reduce_t() = default;
  inline group_reduce_t(
      uint32_t sg_id_,
      uint32_t nbarrier_id,
      uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
  }
  inline void init(
      uint32_t sg_id_ = 0,
      uint32_t nbarrier_id = 0,
      uint32_t slm_base_ = 0) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
  }
  inline void set_slm_base(uint32_t slm_base_ = 0) {
    slm_base = slm_base_;
  }

  inline KERNEL_FUNC xetla_vector<T, N> operator()(
      xetla_vector<T, N * SZ> buffer) {
    local_st_t local_st;
    local_st_payload_t local_st_payload;
    xetla_vector<T, N> ret = sg_reduce(buffer);
    local_st.reg = ret;
    local_st_payload.init(slm_base, N_SG * N, 1, N_SG * N, sg_id * N, 0);
    subgroup::tile_store(local_st, local_st_payload);
    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive();
    nbarrier.wait();
    if constexpr (is_all_reduce) {
      local_ld_t local_ld;
      local_ld_payload_t local_ld_payload(
          slm_base, N_SG * N, 1, N_SG * N, 0, 0);
      subgroup::tile_load(local_ld, local_ld_payload);
      ret = recur_row_reduce<Op, T, N, N_SG>(local_ld.reg);
    } else {
      if (sg_id == 0) {
        local_ld_t local_ld;
        local_ld_payload_t local_ld_payload;
        local_ld_payload.init(slm_base, N_SG * N, 1, N_SG * N, 0, 0);
        subgroup::tile_load(local_ld, local_ld_payload);
        ret = recur_row_reduce<Op, T, N, N_SG>(local_ld.reg);
      }
    }
    return ret;
  }
};

template <
    typename T,
    uint32_t SZ,
    uint32_t N,
    reduce_op Op,
    bool is_all_reduce,
    gpu_arch arch_tag_>
struct group_reduce_t<T, SZ, N, Op, 1, is_all_reduce, arch_tag_> {
  inline group_reduce_t() = default;
  inline group_reduce_t(
      [[maybe_unused]] uint32_t sg_id_,
      [[maybe_unused]] uint32_t nbarrier_id,
      [[maybe_unused]] uint32_t slm_base_) {}
  inline void init(
      [[maybe_unused]] uint32_t sg_id_ = 0,
      [[maybe_unused]] uint32_t nbarrier_id = 0,
      [[maybe_unused]] uint32_t slm_base_ = 0) {}
  inline void set_slm_base([[maybe_unused]] uint32_t slm_base_ = 0) {}
  inline KERNEL_FUNC xetla_vector<T, N> operator()(
      xetla_vector<T, N * SZ> buffer) {
    if constexpr (SZ == 1)
      return buffer;
    auto buffer_2d = buffer.xetla_format<T, N, SZ>();
    xetla_vector<T, N> ret;
#pragma unroll
    for (uint32_t i = 0; i < N; i++) {
      ret[i] = xetla_reduce<T, T, SZ, Op>(buffer_2d.row(i));
    }
    return ret;
  }
};

} // namespace gpu::xetla::group
