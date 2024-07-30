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

#include <group/gemm/common.hpp>

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief Compute policy for xmx engine.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    gpu_arch arch_tag_,
    typename enable = void>
struct compute_policy_default_xmx {};

/// @brief Specialized for Xe architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    gpu_arch arch_tag_>
struct compute_policy_default_xmx<
    compute_attr_,
    perf_tuning_knob_,
    arch_tag_,
    std::enable_if_t<arch_has_xmx<arch_tag_>>> {
  static constexpr gpu_arch arch_tag = arch_tag_;
  static constexpr mma_engine mma_engine = mma_engine::xmx;

  using compute_attr = compute_attr_;
  using dtype_mma_acc = typename compute_attr::dtype_acc;
  using dtype_mma_a = typename compute_attr::dtype_a;
  using dtype_mma_b = typename compute_attr::dtype_b;

  using perf_tuning_knob = perf_tuning_knob_;
  static constexpr int stages = perf_tuning_knob::stages;
  static constexpr int sync_freq = perf_tuning_knob::sync_freq;
  static constexpr int k_stride = perf_tuning_knob::k_stride;

  using mma_attr = mma_attr_t<arch_tag, mma_engine, 16>;

  static constexpr uint32_t block_size_y_a = mma_attr::blk_m_in_elem;
  static constexpr uint32_t block_bytes_x_a = mma_attr::mma_k_in_bytes;
  static constexpr uint32_t block_size_x_a =
      block_bytes_x_a / sizeof(dtype_mma_a);

  static constexpr uint32_t block_size_x_b = mma_attr::mma_n_in_elem;
  static constexpr uint32_t block_size_y_b = block_size_x_a;
  static constexpr uint32_t block_bytes_y_b =
      block_size_x_a * sizeof(dtype_mma_b);
};

/// @brief Compute policy for unaligned shape and xmx engine.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
/// @brief Specialized for Xe architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    gpu_arch arch_tag_>
struct compute_policy_unaligned_xmx : public compute_policy_default_xmx<
                                          compute_attr_,
                                          perf_tuning_knob_,
                                          arch_tag_> {};

/// @brief Compute policy for fpu engine.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    gpu_arch arch_tag_,
    typename enable = void>
struct compute_policy_default_fpu {};

/// @brief Specialized for Xe architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    gpu_arch arch_tag_>
struct compute_policy_default_fpu<
    compute_attr_,
    perf_tuning_knob_,
    arch_tag_,
    std::enable_if_t<arch_has_fpu<arch_tag_>>> {
  static constexpr gpu_arch arch_tag = arch_tag_;
  static constexpr mma_engine mma_engine = mma_engine::fpu;

  using compute_attr = compute_attr_;
  using dtype_mma_acc = typename compute_attr::dtype_acc;
  using dtype_mma_a = typename compute_attr::dtype_a;
  using dtype_mma_b = typename compute_attr::dtype_b;

  using perf_tuning_knob = perf_tuning_knob_;
  static constexpr int stages = perf_tuning_knob::stages;
  static constexpr int sync_freq = perf_tuning_knob::sync_freq;
  static constexpr int k_stride = perf_tuning_knob::k_stride;

  using mma_attr = mma_attr_t<arch_tag, mma_engine, 16>;
  static constexpr uint32_t block_size_y_a = mma_attr::blk_m_in_elem;
  static constexpr uint32_t block_bytes_x_a = mma_attr::blk_k_in_bytes;
  static constexpr uint32_t block_size_x_a =
      block_bytes_x_a / sizeof(dtype_mma_a);

  static constexpr uint32_t block_bytes_x_b = mma_attr::blk_n_in_bytes;
  static constexpr uint32_t block_size_x_b =
      block_bytes_x_b / sizeof(dtype_mma_b);
  static constexpr uint32_t block_size_y_b = block_size_x_a;
};

template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    gpu_arch arch_tag_>
struct compute_policy_unaligned_fpu : public compute_policy_default_fpu<
                                          compute_attr_,
                                          perf_tuning_knob_,
                                          arch_tag_> {};
/// @} xetla_gemm

} // namespace gpu::xetla::group
