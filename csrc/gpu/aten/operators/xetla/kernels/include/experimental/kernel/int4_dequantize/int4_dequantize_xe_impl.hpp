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

#include <experimental/kernel/int4_dequantize/api.hpp>
#include <experimental/kernel/int4_dequantize/config.hpp>

namespace gpu::xetla::kernel {
template <
    typename dtype_qweight_,
    typename dtype_scale_,
    typename dtype_zp_,
    typename dtype_dequant_weight_,
    mem_layout mem_layout_qweight_,
    mem_layout mem_layout_scale_,
    mem_layout mem_layout_zp_,
    mem_layout mem_layout_dequant_weight_,
    quant_info quant_info_,
    typename int4_dequantize_attr_,
    gpu_arch arch_>
struct int4_dequantize_t<
    dtype_qweight_,
    dtype_scale_,
    dtype_zp_,
    dtype_dequant_weight_,
    mem_layout_qweight_,
    mem_layout_scale_,
    mem_layout_zp_,
    mem_layout_dequant_weight_,
    quant_info_,
    int4_dequantize_attr_,
    arch_> {
  static_assert(
      mem_layout_qweight_ == mem_layout::col_major,
      "only support col_major qweight now.");
  static_assert(
      mem_layout_scale_ == mem_layout::col_major,
      "only support col_major scale now.");
  static_assert(
      mem_layout_zp_ == mem_layout::row_major,
      "only support row_major zp now.");
  static_assert(
      mem_layout_dequant_weight_ == mem_layout::row_major,
      "only support row_major dequant_weight now.");

  static constexpr uint32_t dequant_s = quant_info_.dequant_s;
  static constexpr uint32_t pack_ratio = sizeof(dtype_qweight_) * 2;
  static constexpr uint32_t wg_tile_n = int4_dequantize_attr_::wg_tile_n;
  static constexpr uint32_t wg_tile_k = int4_dequantize_attr_::wg_tile_k;
  static constexpr uint32_t sg_tile_n = int4_dequantize_attr_::sg_tile_n;
  static constexpr uint32_t sg_tile_k = int4_dequantize_attr_::sg_tile_k;
  static constexpr uint32_t k_stride = int4_dequantize_attr_::k_stride;

  static_assert(
      wg_tile_n % sg_tile_n == 0,
      "wg_tile_n must be multiple of sg_tile_n");
  static_assert(
      wg_tile_k % sg_tile_k == 0,
      "wg_tile_k must be multiple of sg_tile_k");
  static_assert(
      sg_tile_k % k_stride == 0,
      "sg_tile_k must be multiple of k_stride");

  using mem_desc_qweight_t = mem_desc_t<
      dtype_qweight_,
      mem_layout_qweight_,
      mem_space::global,
      64 / sizeof(dtype_qweight_)>;
  using mem_desc_scale_t = mem_desc_t<
      dtype_scale_,
      mem_layout_scale_,
      mem_space::global,
      64 / sizeof(dtype_scale_)>;
  using mem_desc_zp_t = mem_desc_t<
      dtype_zp_,
      mem_layout_zp_,
      mem_space::global,
      64 / sizeof(dtype_zp_)>;
  using mem_desc_dequant_weight_t = mem_desc_t<
      dtype_dequant_weight_,
      mem_layout_dequant_weight_,
      mem_space::global,
      64 / sizeof(dtype_dequant_weight_)>;

  struct arguments_t {
    uint32_t matrix_k;
    uint32_t matrix_n;
    dtype_qweight_* qweight_base;
    dtype_scale_* scale_base;
    dtype_zp_* zp_base;
    dtype_dequant_weight_* dequant_weight_base;
    uint32_t qweight_ld;
    uint32_t dequant_weight_ld;
    uint32_t scale_ld;
    uint32_t zp_ld;
  };

  static cl::sycl::range<3> get_local_range() {
    uint32_t local_range_k = (wg_tile_k + sg_tile_k - 1) / sg_tile_k;
    uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    XETLA_PRINTF("Local range: {%d, %d, %d}", 1, local_range_k, local_range_n);
    return cl::sycl::range<3>{1, local_range_k, local_range_n};
  };

  static cl::sycl::range<3> get_group_range(
      uint32_t matrix_k,
      uint32_t matrix_n) {
    uint32_t group_range_k = (matrix_k + wg_tile_k - 1) / wg_tile_k;
    uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    XETLA_PRINTF("Group range: {%d, %d, %d}", 1, group_range_k, group_range_n);
    return cl::sycl::range<3>{1, group_range_k, group_range_n};
  };

  static cl::sycl::nd_range<3> get_nd_range(arguments_t& args) {
    cl::sycl::range<3> local_range = get_local_range();
    cl::sycl::range<3> group_range =
        get_group_range(args.matrix_k, args.matrix_n);
    return cl::sycl::nd_range<3>{group_range * local_range, local_range};
  };

  using mat_qweight_tile_desc_t = subgroup::tile_desc_t<
      sg_tile_n, // always N-dim
      k_stride / pack_ratio, // always K-dim
      sg_tile_n, // will be y-tile-dim in col-major qweight.
      k_stride / pack_ratio, // will be x-tile-dim in col-major qweight.
      reg_layout::tiled>;

  using mat_dequant_weight_tile_desc_t = subgroup::
      tile_desc_t<sg_tile_n, k_stride, sg_tile_n, k_stride, reg_layout::tiled>;

  static constexpr uint32_t block_size_y_scale =
      (k_stride + dequant_s - 1) / dequant_s;

  using scale_tile_desc_t = subgroup::tile_desc_t<
      sg_tile_n,
      block_size_y_scale,
      sg_tile_n,
      block_size_y_scale,
      reg_layout::transpose_tiled>;
  using zp_tile_desc_t = subgroup::tile_desc_t<
      (sg_tile_n + pack_ratio - 1) / pack_ratio,
      block_size_y_scale,
      (sg_tile_n + pack_ratio - 1) / pack_ratio,
      block_size_y_scale>;

  using mat_qweight_t =
      subgroup::tile_t<dtype_qweight_, mat_qweight_tile_desc_t>;
  using mat_dequant_weight_t =
      subgroup::tile_t<dtype_dequant_weight_, mat_dequant_weight_tile_desc_t>;
  using scale_t = subgroup::tile_t<dtype_scale_, scale_tile_desc_t>;
  using zp_t = subgroup::tile_t<dtype_zp_, zp_tile_desc_t>;

  // block-wise load, will trade block_size_y as bytes per row block with
  // col-major weight.
  using mat_qweight_payload_t = subgroup::mem_payload_t<
      mem_desc_qweight_t,
      mat_qweight_tile_desc_t,
      subgroup::msg_type_v<mat_qweight_tile_desc_t, mem_desc_qweight_t>,
      arch_>;
  using mat_dequant_weight_payload_t = subgroup::mem_payload_t<
      mem_desc_dequant_weight_t,
      mat_dequant_weight_tile_desc_t,
      subgroup::
          msg_type_v<mat_dequant_weight_tile_desc_t, mem_desc_dequant_weight_t>,
      arch_>;
  using scale_payload_t = subgroup::mem_payload_t<
      mem_desc_scale_t,
      scale_tile_desc_t,
      subgroup::msg_type_v<scale_tile_desc_t, mem_desc_scale_t>,
      arch_>;
  using zp_payload_t = subgroup::mem_payload_t<
      mem_desc_zp_t,
      zp_tile_desc_t,
      subgroup::msg_type_v<zp_tile_desc_t, mem_desc_zp_t>,
      arch_>;
  using dequantize_t = subgroup::dequant_int4_weight_t<
      mat_dequant_weight_t,
      mat_qweight_t,
      scale_t,
      zp_t,
      dequant_s,
      quant_info_.quant_mode>;

  static constexpr uint32_t quant_factor_update_freq =
      (k_stride < dequant_s) ? dequant_s / k_stride : 1;
  __XETLA_API static void call(
      sycl::nd_item<3>& item,
      const arguments_t& args) {
    int wg_id_n = item.get_group(2);
    int wg_id_k = item.get_group(1);
    int sg_id_n = item.get_local_id(2);
    int sg_id_k = item.get_local_id(1);
    int start_k = wg_id_k * wg_tile_k + sg_id_k * sg_tile_k;
    int start_n = wg_id_n * wg_tile_n + sg_id_n * sg_tile_n;
    int start_x_scale = start_n;
    int start_y_scale = start_k / dequant_s;
    int start_x_zp = start_n / pack_ratio;
    int start_y_zp = start_k / dequant_s;

    mem_desc_qweight_t mem_desc_qweight(
        args.qweight_base,
        {start_n + sg_tile_n, // compressed KxN weight width(N)
         start_k + sg_tile_k, // compressed KxN weight height(K)
         args.qweight_ld / pack_ratio}, // compressed weight pitch
        {start_n,
         int(start_k /
             pack_ratio)}); // compressed KxN weight offset_x, offset_y
    mem_desc_dequant_weight_t mem_desc_dequant_weight(
        args.dequant_weight_base,
        {start_n + sg_tile_n, start_k + sg_tile_k, args.dequant_weight_ld},
        {start_n, start_k});
    uint32_t scale_size_y = ((args.matrix_k + dequant_s - 1) / dequant_s);
    mem_desc_scale_t mem_desc_scale(
        args.scale_base,
        {args.matrix_n, scale_size_y, args.scale_ld},
        {start_x_scale, start_y_scale});
    mem_desc_zp_t mem_desc_zp(
        args.zp_base,
        {(args.matrix_n + pack_ratio - 1) / pack_ratio,
         (args.matrix_k + dequant_s - 1) / dequant_s,
         args.zp_ld / pack_ratio},
        {start_x_zp, start_y_zp});
    uint32_t k_dim_loop = sg_tile_k / k_stride;

    mat_qweight_t mat_qweight;
    mat_dequant_weight_t mat_dequant_weight;
    scale_t scale;
    zp_t zp;

    mat_qweight_payload_t mat_qweight_payload(mem_desc_qweight);
    mat_dequant_weight_payload_t mat_dequant_weight_payload(
        mem_desc_dequant_weight);
    scale_payload_t scale_payload(mem_desc_scale);
    zp_payload_t zp_payload(mem_desc_zp);
    typename dequantize_t::arguments_t dequantize_args(start_n, start_k);
    dequantize_t dequantize;
    int tile_k_idx = (start_k + k_stride - 1) / k_stride;
    SW_BARRIER();
#pragma unroll
    for (uint32_t i = 0; i < k_dim_loop; i++) {
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          mat_qweight, mat_qweight_payload);
      subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
          scale, scale_payload);
      if constexpr (quant_info_.quant_mode == quant_mode::I4_ASYM) {
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
            zp, zp_payload);
      }
      tile_k_idx++;
      SW_BARRIER();
      mat_qweight_payload.template update_tdesc<tdesc_update_dir::x_dir>(
          mat_qweight_t::tile_size_y);

      if (tile_k_idx % quant_factor_update_freq == 0) {
        scale_payload.template update_tdesc<tdesc_update_dir::x_dir>(
            scale_t::tile_size_y);
        if constexpr (quant_info_.quant_mode == quant_mode::I4_ASYM) {
          zp_payload.template update_tdesc<tdesc_update_dir::y_dir>(
              zp_t::tile_size_y);
        }
      }
      SW_BARRIER();
      dequantize(mat_dequant_weight, mat_qweight, scale, zp, dequantize_args);
      tile_transpose(mat_dequant_weight);
      subgroup::tile_store(mat_dequant_weight, mat_dequant_weight_payload);
      mat_dequant_weight_payload.template update_tdesc<tdesc_update_dir::y_dir>(
          mat_dequant_weight_t::tile_size_y);
      SW_BARRIER();
    }
  };
};
} // namespace gpu::xetla::kernel
