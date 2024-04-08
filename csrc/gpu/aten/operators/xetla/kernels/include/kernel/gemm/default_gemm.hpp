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

#include <kernel/default_config/common.hpp>
#include <kernel/gemm/common.hpp>
#include <kernel/gemm/dispatch_policy.hpp>
#include <kernel/gemm/gemm_preset.hpp>

namespace gpu::xetla {
namespace kernel {
template <
    typename dtype_a,
    mem_layout mem_layout_a,
    uint32_t alignment_a,
    typename dtype_b,
    mem_layout mem_layout_b,
    uint32_t alignment_b,
    typename dtype_c,
    mem_layout mem_layout_c,
    uint32_t alignment_c,
    typename dtype_acc,
    gpu_arch arch_tag = gpu_arch::XeHpc,
    typename tune_option = dict_t<>>
struct default_gemm_config_t
    : param_adaptor<
          param_adaptor_tag::kernel,
          typename param_optimizer<
              param_optimizer_tag::kernel,
              typename default_param_t<arch_tag>::template update_dict_t<
                  typename tune_option::template update_t<
                      elem_t_t<tune_key::data_type_a, dtype_a>,
                      elem_v_t<tune_key::memory_layout_a, mem_layout_a>,
                      elem_v_t<tune_key::memory_alignment_a, alignment_a>,
                      elem_t_t<tune_key::data_type_b, dtype_b>,
                      elem_v_t<tune_key::memory_layout_b, mem_layout_b>,
                      elem_v_t<tune_key::memory_alignment_b, alignment_b>,
                      elem_t_t<tune_key::data_type_c, dtype_c>,
                      elem_v_t<tune_key::memory_layout_c, mem_layout_c>,
                      elem_v_t<tune_key::memory_alignment_c, alignment_c>,
                      elem_t_t<tune_key::data_type_acc, dtype_acc>,
                      elem_v_t<tune_key::gpu_arch, arch_tag>>>>::type> {};

template <
    typename dtype_a,
    mem_layout mem_layout_a,
    uint32_t alignment_a,
    typename dtype_b,
    mem_layout mem_layout_b,
    uint32_t alignment_b,
    typename dtype_c,
    mem_layout mem_layout_c,
    uint32_t alignment_c,
    typename dtype_acc,
    gpu_arch arch_tag = gpu_arch::XeHpc,
    typename tune_option = dict_t<>>
using default_gemm_t = typename default_gemm_config_t<
    dtype_a,
    mem_layout_a,
    alignment_a,
    dtype_b,
    mem_layout_b,
    alignment_b,
    dtype_c,
    mem_layout_c,
    alignment_c,
    dtype_acc,
    arch_tag,
    tune_option>::type;
} // namespace kernel

template <typename dict_t_>
struct param_optimizer<param_optimizer_tag::kernel, dict_t_> {
  static constexpr bool use_rule =
      (dict_t_::impl::template find_elem_index<
           tune_key::param_optimizer_type> != dict_t_::impl::key_not_found) &&
      (dict_t_::template find_elem_v<tune_key::param_optimizer_type> ==
       tune_key_value::param_optimizer_decision_tree);
  static constexpr auto arch_tag =
      (dict_t_::impl::template find_elem_index<tune_key::gpu_arch> !=
       dict_t_::impl::key_not_found)
      ? dict_t_::template find_elem_v<tune_key::gpu_arch>
      : gpu_arch::XeHpc;
  static constexpr auto optimizer_level =
      dict_t_::template find_elem_v<tune_key::param_optimizer_level>;
  using type = typename std::conditional<
      use_rule,
      decision_tree_optimizer<
          param_optimizer_tag::kernel,
          dict_t_,
          optimizer_level>,
      dummy_optimizer<
          param_optimizer_tag::kernel,
          dict_t_,
          kernel::param_kslicing_g1l1_t<arch_tag>,
          kernel::param_kslicing_g2l1_t<arch_tag>,
          kernel::param_kslicing_g1l2_t<arch_tag>>>::type::type;
};

template <typename dict_t_>
struct param_adaptor<param_adaptor_tag::kernel, dict_t_>
    : param_adaptor_base<dict_t_> {
  using param = typename dict_t_::template update_t<
      elem_v_t<tune_key::memory_space_a, mem_space::global>,
      elem_v_t<tune_key::memory_space_b, mem_space::global>,
      elem_v_t<tune_key::memory_space_c, mem_space::global>>;
  using base_t = param_adaptor_base<param>;

  using gemm_t =
      typename param_adaptor<param_adaptor_tag::work_group_gemm, param>::type;
  using epilogue_t =
      typename param_adaptor<param_adaptor_tag::work_group_epilogue, param>::
          type;

  using group_swizzle = typename param::template find_elem_t<
      tune_key::group_swizzle_policy>::type;

  static constexpr auto dispatch_policy_tag =
      param::template find_elem_v<tune_key::dispatch_policy>;
  static constexpr int num_global_splitk =
      param::template find_elem_v<tune_key::global_kslicing_ratio>;
  static constexpr int num_local_splitk =
      param::template find_elem_v<tune_key::local_kslicing_ratio>;
  using dispatch_policy = typename dict_t<
      elem_t_t<
          tune_key_value::dispatch_policy_default,
          kernel::dispatch_policy_default<group_swizzle>>,
      elem_t_t<
          tune_key_value::dispatch_policy_kslicing,
          kernel::dispatch_policy_kslicing<
              group_swizzle,
              num_global_splitk,
              num_local_splitk>>,
      elem_t_t<
          tune_key_value::dispatch_policy_stream_k,
          kernel::dispatch_policy_stream_k<base_t::gpu_arch_tag>>

      >::template find_elem_t<dispatch_policy_tag>::type;

  using type = kernel::gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;
};

namespace group {
template <
    typename dtype_a,
    mem_layout mem_layout_a,
    uint32_t alignment_a,
    mem_space mem_space_a,
    typename dtype_b,
    mem_layout mem_layout_b,
    uint32_t alignment_b,
    mem_space mem_space_b,
    typename dtype_acc,
    typename wg_shape,
    uint32_t wg_tile_k,
    gpu_arch arch_tag = gpu_arch::XeHpc,
    typename tune_option = dict_t<>>
struct default_gemm_selector_config_t
    : param_adaptor<
          param_adaptor_tag::work_group_gemm,
          typename param_optimizer<
              param_optimizer_tag::work_group,
              typename default_param_t<arch_tag>::template update_dict_t<
                  typename tune_option::template update_t<
                      elem_t_t<tune_key::data_type_a, dtype_a>,
                      elem_v_t<tune_key::memory_layout_a, mem_layout_a>,
                      elem_v_t<tune_key::memory_alignment_a, alignment_a>,
                      elem_v_t<tune_key::memory_space_a, mem_space_a>,
                      elem_t_t<tune_key::data_type_b, dtype_b>,
                      elem_v_t<tune_key::memory_layout_b, mem_layout_b>,
                      elem_v_t<tune_key::memory_alignment_b, alignment_b>,
                      elem_v_t<tune_key::memory_space_b, mem_space_b>,
                      elem_t_t<tune_key::data_type_acc, dtype_acc>,
                      elem_t_t<tune_key::wg_tile_shape, wg_shape>,
                      elem_v_t<tune_key::wg_tile_k, wg_tile_k>,
                      elem_v_t<tune_key::gpu_arch, arch_tag>>>>::type> {};

template <
    typename dtype_a,
    mem_layout mem_layout_a,
    uint32_t alignment_a,
    mem_space mem_space_a,
    typename dtype_b,
    mem_layout mem_layout_b,
    uint32_t alignment_b,
    mem_space mem_space_b,
    typename dtype_acc,
    typename wg_shape,
    uint32_t wg_tile_k,
    gpu_arch arch_tag = gpu_arch::XeHpc,
    typename tune_option = dict_t<>>
using default_gemm_selector_t = typename default_gemm_selector_config_t<
    dtype_a,
    mem_layout_a,
    alignment_a,
    mem_space_a,
    dtype_b,
    mem_layout_b,
    alignment_b,
    mem_space_b,
    dtype_acc,
    wg_shape,
    wg_tile_k,
    arch_tag,
    tune_option>::type;

template <
    typename dtype_c,
    mem_layout mem_layout_c,
    uint32_t alignment_c,
    mem_space mem_space_c,
    typename wg_shape,
    uint32_t wg_tile_k,
    gpu_arch arch_tag = gpu_arch::XeHpc,
    typename tune_option = dict_t<>>
struct default_epilogue_selector_config_t
    : param_adaptor<
          param_adaptor_tag::work_group_epilogue,
          typename param_optimizer<
              param_optimizer_tag::work_group,
              typename default_param_t<arch_tag>::template update_dict_t<
                  typename tune_option::template update_t<
                      elem_t_t<tune_key::data_type_c, dtype_c>,
                      elem_v_t<tune_key::memory_layout_c, mem_layout_c>,
                      elem_v_t<tune_key::memory_alignment_c, alignment_c>,
                      elem_v_t<tune_key::memory_space_c, mem_space_c>,
                      elem_t_t<tune_key::wg_tile_shape, wg_shape>,
                      elem_v_t<tune_key::wg_tile_k, wg_tile_k>,
                      elem_v_t<tune_key::gpu_arch, arch_tag>>>>::type> {};

template <
    typename dtype_c,
    mem_layout mem_layout_c,
    uint32_t alignment_c,
    mem_space mem_space_c,
    typename wg_shape,
    uint32_t wg_tile_k,
    gpu_arch arch_tag = gpu_arch::XeHpc,
    typename tune_option = dict_t<>>
using default_epilogue_selector_t = typename default_epilogue_selector_config_t<
    dtype_c,
    mem_layout_c,
    alignment_c,
    mem_space_c,
    wg_shape,
    wg_tile_k,
    arch_tag,
    tune_option>::type;
} // namespace group

template <typename dict_t_>
struct param_optimizer<param_optimizer_tag::work_group, dict_t_> {
  static constexpr bool use_rule =
      (dict_t_::impl::template find_elem_index<
           tune_key::param_optimizer_type> != dict_t_::impl::key_not_found) &&
      (dict_t_::template find_elem_v<tune_key::param_optimizer_type> ==
       tune_key_value::param_optimizer_decision_tree);
  static constexpr auto optimizer_level =
      dict_t_::template find_elem_v<tune_key::param_optimizer_level>;
  static constexpr auto arch_tag =
      (dict_t_::impl::template find_elem_index<tune_key::gpu_arch> !=
       dict_t_::impl::key_not_found)
      ? dict_t_::template find_elem_v<tune_key::gpu_arch>
      : gpu_arch::XeHpc;
  using type = typename std::conditional<
      use_rule,
      decision_tree_optimizer<
          param_optimizer_tag::work_group,
          dict_t_,
          optimizer_level>,
      dummy_optimizer<
          param_optimizer_tag::work_group,
          dict_t_,
          group::param_dict1_wg_t<arch_tag>>>::type::type;
};

template <typename dict_t_>
struct param_adaptor<param_adaptor_tag::work_group_gemm, dict_t_>
    : param_adaptor_base<dict_t_> {
  using param = dict_t_;
  using base_t = param_adaptor_base<param>;

  using dtype_a =
      typename param::template find_elem_t<tune_key::data_type_a>::type;
  using dtype_b =
      typename param::template find_elem_t<tune_key::data_type_b>::type;
  static constexpr auto mem_layout_a =
      param::template find_elem_v<tune_key::memory_layout_a>;
  static constexpr auto mem_layout_b =
      param::template find_elem_v<tune_key::memory_layout_b>;
  static constexpr auto mem_space_a =
      param::template find_elem_v<tune_key::memory_space_a>;
  static constexpr auto mem_space_b =
      param::template find_elem_v<tune_key::memory_space_b>;
  static constexpr auto mem_alignment_a =
      param::template find_elem_v<tune_key::memory_alignment_a>;
  static constexpr auto mem_alignment_b =
      param::template find_elem_v<tune_key::memory_alignment_b>;

  using compute_attr =
      group::compute_attr_t<dtype_a, dtype_b, typename base_t::dtype_acc>;

  using perf_tuning_knob = group::perf_tuning_knob_t<
      base_t::wg_tile_k,
      base_t::prefetch_distance,
      base_t::periodic_sync_interval>;

  // specific the computation, performance tuning and computation core
  using compute_policy = typename dict_t<
      elem_t_t<
          mma_engine::xmx,
          typename std::conditional<
              (group::detail::check_2d_block_pitch_alignment<
                  dtype_a,
                  dtype_b,
                  mem_alignment_a,
                  mem_alignment_b,
                  base_t::gpu_arch_tag>::value),
              group::compute_policy_default_xmx<
                  compute_attr,
                  perf_tuning_knob,
                  base_t::gpu_arch_tag>,
              group::compute_policy_unaligned_xmx<
                  compute_attr,
                  perf_tuning_knob,
                  base_t::gpu_arch_tag>>::type>,
      elem_t_t<
          mma_engine::fpu,
          typename std::conditional<
              (group::detail::check_2d_block_pitch_alignment<
                  dtype_a,
                  dtype_b,
                  mem_alignment_a,
                  mem_alignment_b,
                  base_t::gpu_arch_tag>::value),
              group::compute_policy_default_fpu<
                  compute_attr,
                  perf_tuning_knob,
                  base_t::gpu_arch_tag>,
              void>::type>>::template find_elem_t<base_t::mma_engine_tag>::type;

  using mem_desc_input_a =
      mem_desc_t<dtype_a, mem_layout_a, mem_space_a, mem_alignment_a>;
  using mem_desc_input_b =
      mem_desc_t<dtype_b, mem_layout_b, mem_space_b, mem_alignment_b>;

  static constexpr auto pre_processing_tag =
      param::template find_elem_v<tune_key::pre_processing>;
  using pre_processing = typename std::conditional<
      (pre_processing_tag == tune_key_value::pre_processing_mata_neg_filter),
      group::pre_processing_matA_neg_filter_t<
          typename base_t::tile_shape,
          base_t::gpu_arch_tag>,
      group::pre_processing_default_t<
          typename base_t::tile_shape,
          base_t::gpu_arch_tag>>::type;

  using gemm_t = group::gemm_t<
      compute_policy,
      typename base_t::tile_shape,
      mem_desc_input_a,
      mem_desc_input_b,
      pre_processing>;

  using type = gemm_t;
};

template <typename dict_t_>
struct param_adaptor<param_adaptor_tag::work_group_epilogue, dict_t_> {
  using param = dict_t_;
  using base_t = param_adaptor_base<dict_t_>;

  using dtype_c =
      typename param::template find_elem_t<tune_key::data_type_c>::type;
  static constexpr auto mem_layout_c =
      param::template find_elem_v<tune_key::memory_layout_c>;
  static constexpr auto mem_alignment_c =
      param::template find_elem_v<tune_key::memory_alignment_c>;
  static constexpr auto mem_space_c =
      param::template find_elem_v<tune_key::memory_space_c>;

  using epilogue_policy =
      typename param::template find_elem_t<tune_key::epilogue_policy>::type;

  using epilogue_t = group::epilogue_t<
      epilogue_policy,
      typename base_t::tile_shape,
      mem_desc_t<dtype_c, mem_layout_c, mem_space_c, mem_alignment_c>>;

  using type = epilogue_t;
};
} // namespace gpu::xetla
