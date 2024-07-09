/*******************************************************************************
 * Copyright (c) 2023-2024 Intel Corporation
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

#include <experimental/kernel/layer_norm/common.hpp>

namespace gpu::xetla::kernel {

/// @brief Sets up attribute of the layer norm.
///
/// @tparam wg_tile_x_ Is the num of cols processed by one workgroup.
/// @tparam wg_tile_y_ Is the num of rows processed by one workgroup.
/// @tparam sg_tile_x_ Is the num of cols processed by one subgroup.
/// @tparam sg_tile_y_ Is the num of rows processed by one subgroup.
/// @tparam load_block_size_ Is the size of block when load x dimenstion.
/// kernels have spills.
template <
    uint32_t wg_tile_x_,
    uint32_t wg_tile_y_,
    uint32_t sg_tile_x_,
    uint32_t sg_tile_y_,
    uint32_t load_block_size_>
struct col_major_shuf_attr_t {
  static constexpr uint32_t wg_tile_x = wg_tile_x_;
  static constexpr uint32_t wg_tile_y = wg_tile_y_;
  static constexpr uint32_t sg_tile_x = sg_tile_x_;
  static constexpr uint32_t sg_tile_y = sg_tile_y_;
  static constexpr uint32_t load_block_size = load_block_size_;

  static_assert(
      wg_tile_x % sg_tile_x == 0,
      "Current design we don't enable the boundary check");
  static_assert(
      sg_tile_x % load_block_size == 0 && sg_tile_x >= load_block_size,
      "Current design we don't enable the boundary check on chunking "
      "mechanism");
};

} // namespace gpu::xetla::kernel
