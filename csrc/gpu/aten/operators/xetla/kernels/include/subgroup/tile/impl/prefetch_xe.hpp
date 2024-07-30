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
#include <subgroup/tile/impl/op_function.hpp>
#include <subgroup/tile/impl/payload_xe.hpp>

namespace gpu::xetla::subgroup {
namespace detail {

template <typename payload_t, typename = void>
struct check_prefetch_type;
template <typename payload_t>
struct check_prefetch_type<
    payload_t,
    std::enable_if_t<payload_t::memory_space == mem_space::local>> {
  static constexpr bool is_global_2d = false;
  static constexpr bool is_global_block_1d = false;
  static constexpr bool is_global_unaligned_2d = false;
  static constexpr bool is_local = true;
};
template <typename payload_t>
struct check_prefetch_type<
    payload_t,
    std::enable_if_t<payload_t::memory_space == mem_space::global>> {
  static constexpr bool is_global_2d =
      payload_t::message_type == msg_type::block_2d;
  static constexpr bool is_global_block_1d =
      payload_t::message_type == msg_type::block_1d;
  static constexpr bool is_global_unaligned_2d =
      payload_t::message_type == msg_type::unaligned_2d;
  static constexpr bool is_local = false;
};

} // namespace detail

/// @brief Is prefetch data func, which data located in global memory is
/// prefetched to cache, where has higher bandwidth. e.g. In gemm, prefetch next
/// iteration data for mma consumption. This func is specicalized for block 2d
/// scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_prefetch_type<payload_t>::is_global_2d &&
    arch_has_2d_load_store<payload_t::arch_tag>>
tile_prefetch(payload_t& payload) {
  using dtype = typename payload_t::dtype;
  static constexpr uint32_t num_tdesc = payload_t::num_tdesc;
  auto tdesc_2d =
      payload.tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();

#pragma unroll
  for (uint32_t i = 0; i < num_tdesc; i++) {
    xetla_tprefetch_global<dtype, L1, L2, payload_t::arch_tag>(tdesc_2d.row(i));
  }
}

/// @brief Is prefetch data func, which data located in global memory is
/// prefetched to cache, where has higher bandwidth. e.g. In gemm, prefetch next
/// iteration data for mma consumption. This func is specicalized for block 1d
/// scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_prefetch_type<payload_t>::is_global_2d &&
    !arch_has_2d_load_store<payload_t::arch_tag>>
tile_prefetch(payload_t& payload) {
  using dtype = typename payload_t::dtype;
  using tile_desc = typename payload_t::tile_desc;
  using prefetch_dtype = typename payload_t::prefetch_dtype;
  constexpr uint32_t num_channel = payload_t::num_channel;
#pragma unroll
  for (uint32_t i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y;
       i++) {
    uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
    for (uint32_t j = 0; j < tile_desc::num_block_x; j++) {
      uint32_t offset_x = j * tile_desc::block_size_x;
      // #pragma unroll
      //       for (uint32_t sub_block_y = 0; sub_block_y <
      //       tile_desc::block_size_y;
      //            sub_block_y += num_channel) {
      uint32_t address_offset = payload_t::mem_transpose
          ? offset_x * payload.pitch_in_bytes + (offset_y + 0) * sizeof(dtype)
          : offset_x * sizeof(dtype) + (offset_y + 0) * payload.pitch_in_bytes;

      xetla_prefetch_global<
          prefetch_dtype,
          num_channel * payload_t::vector_size,
          payload_t::vector_size,
          L1,
          L2>(
          payload.base_ptr,
          payload.channel_offset + payload.base_offset + address_offset);
      //   }
    }
  }
}

/// @brief Is prefetch data func, which data located in global memory is
/// prefetched to cache, where has higher bandwidth. e.g. In gemm, prefetch next
/// iteration data for mma consumption. This func is specicalized for block 1d
/// scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_prefetch_type<payload_t>::is_global_block_1d>
tile_prefetch(payload_t& payload) {
  using dtype = typename payload_t::dtype;
  using prefetch_dtype = typename payload_t::prefetch_dtype;
  using tile_desc = typename payload_t::tile_desc;
  constexpr uint32_t prefetch_len = payload_t::mem_transpose
      ? tile_desc::tile_size_y
      : tile_desc::tile_size_x;
  constexpr uint32_t max_prefetch_in_bytes =
      load_store_attr_t<msg_type::block_1d, payload_t::arch_tag>::
          max_prefetch_vec_len;
  constexpr uint32_t max_prefetch_in_elems =
      max_prefetch_in_bytes / sizeof(dtype);

  static constexpr uint32_t prefetch_iter_steps =
      prefetch_len / max_prefetch_in_elems;
  if constexpr (prefetch_len >= max_prefetch_in_elems) {
#pragma unroll
    for (uint32_t i = 0; i < prefetch_iter_steps; i++) {
      uint32_t byte_offset = i * max_prefetch_in_bytes;
      xetla_prefetch_global<
          prefetch_dtype,
          max_prefetch_in_bytes / sizeof(prefetch_dtype),
          L1,
          L2>(payload.base_ptr, payload.base_offset + byte_offset);
    }
  }
  constexpr uint32_t tail_len =
      prefetch_len % max_prefetch_in_elems * sizeof(dtype);

  uint32_t tail_offset = prefetch_iter_steps * max_prefetch_in_bytes;
  detail::process_1d_tail<
      tail_len,
      (max_prefetch_in_bytes >> 1),
      L1,
      L2,
      payload_t>(payload, tail_offset);
}

/// @brief Is prefetch data func.
/// Current shared local memory prefetch is not supported yet. Only used to keep
/// the consistency with global prefetch.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename payload_t>
__XETLA_API
    typename std::enable_if_t<detail::check_prefetch_type<payload_t>::is_local>
    tile_prefetch([[maybe_unused]] payload_t& payload) {}

} // namespace gpu::xetla::subgroup