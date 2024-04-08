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

namespace gpu::xetla::group {

enum quant_mode : uint8_t { S4_ASYM, S4_FULLRANGE_NO_ZP };

/// @brief Compute policy for unaligned shape and xmx engine.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename dtype_scale_,
    typename dtype_zero_pt_,
    quant_mode quant_type_,
    int dequant_s_,
    mma_engine mma_engine_ = mma_engine::xmx,
    gpu_arch arch_tag_ = gpu_arch::XeHpc,
    typename enable = void>
struct compute_policy_int4_dequantize {};

/// @brief Specialized for XeHpc and XeHpg architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename dtype_scale_,
    typename dtype_zero_pt_,
    quant_mode quant_type_,
    int dequant_s_,
    mma_engine mma_engine_,
    gpu_arch arch_tag_>
struct compute_policy_int4_dequantize<
    compute_attr_,
    perf_tuning_knob_,
    dtype_scale_,
    dtype_zero_pt_,
    quant_type_,
    dequant_s_,
    mma_engine_,
    arch_tag_,
    std::enable_if_t<(arch_tag_ <= gpu_arch::XeHpc)>> {
  using compute_attr = compute_attr_;
  using perf_tuning_knob = perf_tuning_knob_;
  static constexpr int k_stride = perf_tuning_knob::k_stride;
  static constexpr int stages = perf_tuning_knob::stages;
  static constexpr int sync_freq = perf_tuning_knob::sync_freq;
  static constexpr mma_engine mma_engine = mma_engine_;
  static constexpr gpu_arch arch_tag = arch_tag_;

  static_assert(
      !(mma_engine == mma_engine::xmx && arch_tag == gpu_arch::XeLpg),
      "XeLpg does not support xmx");

  using dtype_mma_acc = typename compute_attr::dtype_acc;
  using dtype_mma_a = typename compute_attr::dtype_a;
  using dtype_mma_b = typename compute_attr::dtype_b;

  static constexpr uint32_t block_bytes_x_a = 32;
  static constexpr uint32_t block_size_y_a = 16;

  static constexpr bool is_int4_matB_policy = true;

  static constexpr uint32_t block_size_x_b =
      arch_attr_t<arch_tag>::mma_attr::mma_n_in_elem;
  static constexpr uint32_t block_bytes_y_b = 32;
  static_assert(
      block_bytes_x_a == block_bytes_y_b,
      "mat_a x need to match with mat_b y");

  static constexpr uint32_t dequant_s = dequant_s_;
  static_assert(
      (dequant_s % (32 / sizeof(dtype_mma_b))) == 0,
      "dequant_s should be a multiply of 32B");
  using dtype_scale = dtype_scale_;
  using dtype_zero_pt = dtype_zero_pt_;
  static constexpr quant_mode quant_type = quant_type_;
};

} // namespace gpu::xetla::group
