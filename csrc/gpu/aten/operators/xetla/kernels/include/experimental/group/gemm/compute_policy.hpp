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
/// @brief Compute policy for int4 dequant gemm.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename dtype_scale_,
    typename dtype_zero_pt_,
    quant_info quant_info_,
    mma_engine mma_engine_ = mma_engine::xmx,
    gpu_arch arch_tag_ = gpu_arch::XeHpc,
    typename enable = void>
struct compute_policy_int4_dequantize {};

/// @brief Specialized for xmx engine.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename dtype_scale_,
    typename dtype_zero_pt_,
    quant_info quant_info_,
    mma_engine mma_engine_,
    gpu_arch arch_tag_>
struct compute_policy_int4_dequantize<
    compute_attr_,
    perf_tuning_knob_,
    dtype_scale_,
    dtype_zero_pt_,
    quant_info_,
    mma_engine_,
    arch_tag_,
    std::enable_if_t<mma_engine_ == mma_engine::xmx>> {
  using compute_attr = compute_attr_;
  using dtype_mma_acc = typename compute_attr::dtype_acc;
  using dtype_mma_a = typename compute_attr::dtype_a;
  using dtype_mma_b = typename compute_attr::dtype_b;

  using perf_tuning_knob = perf_tuning_knob_;
  static constexpr int stages = perf_tuning_knob::stages;
  static constexpr int sync_freq = perf_tuning_knob::sync_freq;
  static constexpr int k_stride = perf_tuning_knob::k_stride;
  static constexpr mma_engine mma_engine = mma_engine_;
  static constexpr gpu_arch arch_tag = arch_tag_;

  static_assert(arch_has_xmx<arch_tag>, "XeLpg does not support xmx");

  static constexpr bool is_int4_matB_policy = true;

  static constexpr uint32_t dequant_s = quant_info_.dequant_s;
  static_assert(
      (dequant_s % (32 / sizeof(dtype_mma_b))) == 0,
      "dequant_s should be a multiply of 32B");
  using dtype_scale = dtype_scale_;
  using dtype_zero_pt = dtype_zero_pt_;
  static constexpr quant_mode quant_mode = quant_info_.quant_mode;

  static constexpr uint32_t block_size_y_a = 16;
  using mma_attr = mma_attr_t<arch_tag_, block_size_y_a>;
  static constexpr uint32_t block_bytes_x_a = mma_attr::mma_k_in_bytes;
  static constexpr uint32_t block_size_x_a =
      block_bytes_x_a / sizeof(dtype_mma_a);
  static constexpr uint32_t block_size_x_b = mma_attr::mma_n_in_elem;
  static constexpr uint32_t block_bytes_y_b = mma_attr::mma_k_in_bytes;
  static constexpr uint32_t block_size_y_b =
      block_bytes_y_b / sizeof(dtype_mma_b);

  static_assert(
      block_bytes_x_a == block_bytes_y_b,
      "mat_a x need to match with mat_b y");
};

/// @brief Specialized for fpu engine.
template <
    typename compute_attr_,
    typename perf_tuning_knob_,
    typename dtype_scale_,
    typename dtype_zero_pt_,
    quant_info quant_info_,
    mma_engine mma_engine_,
    gpu_arch arch_tag_>
struct compute_policy_int4_dequantize<
    compute_attr_,
    perf_tuning_knob_,
    dtype_scale_,
    dtype_zero_pt_,
    quant_info_,
    mma_engine_,
    arch_tag_,
    std::enable_if_t<mma_engine_ == mma_engine::fpu>> {
  using compute_attr = compute_attr_;
  using dtype_mma_acc = typename compute_attr::dtype_acc;
  using dtype_mma_a = typename compute_attr::dtype_a;
  using dtype_mma_b = typename compute_attr::dtype_b;

  using perf_tuning_knob = perf_tuning_knob_;
  static constexpr int stages = perf_tuning_knob::stages;
  static constexpr int sync_freq = perf_tuning_knob::sync_freq;
  static constexpr int k_stride = perf_tuning_knob::k_stride;
  static constexpr mma_engine mma_engine = mma_engine_;
  static constexpr gpu_arch arch_tag = arch_tag_;

  static constexpr bool is_int4_matB_policy = true;

  static constexpr uint32_t dequant_s = quant_info_.dequant_s;
  static_assert(
      (dequant_s % (32 / sizeof(dtype_mma_b))) == 0,
      "dequant_s should be a multiply of 32B");
  using dtype_scale = dtype_scale_;
  using dtype_zero_pt = dtype_zero_pt_;
  static constexpr quant_mode quant_mode = quant_info_.quant_mode;
  static constexpr bool is_col_major_b =
      quant_info_.weight_mem_layout == mem_layout::col_major;

  static constexpr uint32_t block_size_y_a = is_col_major_b ? 8 : 16;
  static constexpr uint32_t block_bytes_x_a = is_col_major_b ? 128 : 32;
  static constexpr uint32_t block_size_x_a =
      block_bytes_x_a / sizeof(dtype_mma_a);
  static constexpr uint32_t block_size_x_b = is_col_major_b ? 1 : 32;
  static constexpr uint32_t block_bytes_y_b = is_col_major_b ? 128 : 32;
  static constexpr uint32_t block_size_y_b =
      block_bytes_y_b / sizeof(dtype_mma_b);

  static_assert(
      block_bytes_x_a == block_bytes_y_b,
      "mat_a x need to match with mat_b y");
};

} // namespace gpu::xetla::group
