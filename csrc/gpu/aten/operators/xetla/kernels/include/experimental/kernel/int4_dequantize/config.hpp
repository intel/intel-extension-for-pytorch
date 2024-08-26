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

#include <common/common.hpp>
#include <group/group.hpp>
#include <subgroup/subgroup.hpp>

namespace gpu::xetla::kernel {

/// @brief Sets up attribute of the int4 dequantize.
///
/// @tparam wg_tile_n_ Is the N-dim of KxN weight processed by one workgroup.
/// @tparam wg_tile_k_ Is the K-dim of KxN weight processed by one workgroup.
/// @tparam sg_tile_n_ Is the N-dim of KxN weight processed by one subgroup.
/// @tparam sg_tile_k_ Is the K-dim of KxN weight processed by one subgroup.
/// @tparam load_block_size_ Is the size of block when load x dimenstion.
/// kernels have spills.
template <
    uint32_t wg_tile_n_,
    uint32_t wg_tile_k_,
    uint32_t sg_tile_n_,
    uint32_t sg_tile_k_,
    uint32_t k_stride_>
struct int4_dequantize_attr_t {
  static constexpr uint32_t wg_tile_n = wg_tile_n_;
  static constexpr uint32_t wg_tile_k = wg_tile_k_;
  static constexpr uint32_t sg_tile_n = sg_tile_n_;
  static constexpr uint32_t sg_tile_k = sg_tile_k_;
  static constexpr uint32_t k_stride = k_stride_;
};

} // namespace gpu::xetla::kernel
