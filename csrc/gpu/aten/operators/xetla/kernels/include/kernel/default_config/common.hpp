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

#pragma once

#include <common/common.hpp>
#include <group/group.hpp>
#include <subgroup/subgroup.hpp>

namespace gpu::xetla {

// enum

enum class tune_key : uint8_t {
  data_type_a,
  memory_layout_a,
  memory_alignment_a,
  memory_space_a,
  data_type_b,
  memory_layout_b,
  memory_alignment_b,
  memory_space_b,
  data_type_c,
  memory_layout_c,
  memory_alignment_c,
  memory_space_c,
  data_type_acc,
  global_kslicing_ratio,
  local_kslicing_ratio,
  wg_tile_shape,
  wg_tile_k,
  sg_tile_shape,
  pre_processing,
  prefetch_distance,
  periodic_sync_interval,
  mma_engine,
  gpu_arch,
  epilogue_policy,
  dispatch_policy,
  group_swizzle_policy,
  param_optimizer_type,
  param_optimizer_level,
  source_location
};
template <typename T>
using data_type_a_t =
    typename T::template find_elem_t<tune_key::data_type_a>::type;
template <typename T>
using data_type_b_t =
    typename T::template find_elem_t<tune_key::data_type_b>::type;
template <typename T>
using data_type_c_t =
    typename T::template find_elem_t<tune_key::data_type_c>::type;
template <typename T>
constexpr auto memory_layout_a_v =
    T::template find_elem_v<tune_key::memory_layout_a>;
template <typename T>
constexpr auto memory_alignment_a_v =
    T::template find_elem_v<tune_key::memory_alignment_a>;
template <typename T>
constexpr auto memory_layout_b_v =
    T::template find_elem_v<tune_key::memory_layout_b>;
template <typename T>
constexpr auto memory_alignment_b_v =
    T::template find_elem_v<tune_key::memory_alignment_b>;
template <typename T>
constexpr auto memory_layout_c_v =
    T::template find_elem_v<tune_key::memory_layout_c>;
template <typename T>
constexpr auto memory_alignment_c_v =
    T::template find_elem_v<tune_key::memory_alignment_c>;
template <typename T>
constexpr auto gpu_arch_v = T::template find_elem_v<tune_key::gpu_arch>;

enum class tune_key_value : uint8_t {
  pre_processing_default,
  pre_processing_mata_neg_filter,
  dispatch_policy_default,
  dispatch_policy_kslicing,
  dispatch_policy_stream_k,
  param_optimizer_dummy,
  param_optimizer_decision_tree
};

// parameter optimizer

enum class param_optimizer_tag : uint8_t { kernel, work_group };
// optimizer_level (currently only useful with param_optimizer_decision_tree)
enum class param_optimizer_level : uint8_t {
  full, // optimize all available options
  keep_shape, // optimize all except keeping the original wg/sg tile shape
};

template <param_optimizer_tag tag_, typename dict_t_>
struct param_optimizer;

struct param_optimizer_base {
  template <typename T, typename U>
  static constexpr bool valid_attribute_v =
      std::is_same_v<data_type_a_t<T>, data_type_a_t<U>> //
          && memory_layout_a_v<T> ==
      memory_layout_a_v<U> //
          && memory_alignment_a_v<T> ==
      memory_alignment_a_v<U> //
          && std::is_same_v<data_type_b_t<T>, data_type_b_t<U>> //
              && memory_layout_b_v<T> ==
      memory_layout_b_v<U> //
          && memory_alignment_b_v<T> ==
      memory_alignment_b_v<U> //
          && std::is_same_v<data_type_c_t<T>, data_type_c_t<U>> //
              && memory_layout_c_v<T> ==
      memory_layout_c_v<U> //
          && memory_alignment_c_v<T> ==
      memory_alignment_c_v<U> //
          && gpu_arch_v<T> == gpu_arch_v<U>;
};

// parameter adaptor

enum class param_adaptor_tag : uint8_t {
  kernel,
  work_group_gemm,
  work_group_epilogue
};

template <param_adaptor_tag tag_, typename dict_t_>
struct param_adaptor;

template <typename dict_t_>
struct param_adaptor_base {
  using dtype_acc =
      typename dict_t_::template find_elem_t<tune_key::data_type_acc>::type;
  using wg_tile_shape =
      typename dict_t_::template find_elem_t<tune_key::wg_tile_shape>::type;
  static constexpr uint32_t wg_tile_n = wg_tile_shape::template dim<0>();
  static constexpr uint32_t wg_tile_m = wg_tile_shape::template dim<1>();
  static constexpr uint32_t wg_tile_k =
      dict_t_::template find_elem_v<tune_key::wg_tile_k>;
  using sg_tile_shape =
      typename dict_t_::template find_elem_t<tune_key::sg_tile_shape>::type;
  static constexpr uint32_t sg_tile_n = sg_tile_shape::template dim<0>();
  static constexpr uint32_t sg_tile_m = sg_tile_shape::template dim<1>();
  static constexpr uint32_t prefetch_distance =
      dict_t_::template find_elem_v<tune_key::prefetch_distance>;
  static constexpr uint32_t periodic_sync_interval =
      dict_t_::template find_elem_v<tune_key::periodic_sync_interval>;
  static constexpr auto mma_engine_tag =
      dict_t_::template find_elem_v<tune_key::mma_engine>;
  static constexpr auto gpu_arch_tag =
      dict_t_::template find_elem_v<tune_key::gpu_arch>;

  // Org the compute shape for sub-matrix
  using tile_shape = group::tile_shape_t<
      wg_tile_n, // workgroup size in dim0
      wg_tile_m, //	workgroup size in dim1
      sg_tile_n, //	subgroup size in dim0
      sg_tile_m>; //	subgroup size in dim1
};

} // namespace gpu::xetla

#include <kernel/default_config/decision_tree_policy.hpp>
#include <kernel/default_config/dummy_policy.hpp>
