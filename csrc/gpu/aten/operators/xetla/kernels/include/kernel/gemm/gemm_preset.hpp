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

#include <kernel/default_config/common.hpp>
#include <kernel/gemm/common.hpp>

namespace gpu::xetla {
namespace detail {
using param_dtype_bf16_bf16_bf16 = dict_t<
    elem_t_t<tune_key::data_type_a, bf16>,
    elem_t_t<tune_key::data_type_b, bf16>,
    elem_t_t<tune_key::data_type_c, bf16>>;

using param_memalignment_8_8_8 = dict_t<
    elem_v_t<tune_key::memory_alignment_a, 8UL, uint32_t>,
    elem_v_t<tune_key::memory_alignment_b, 8UL, uint32_t>,
    elem_v_t<tune_key::memory_alignment_c, 8UL, uint32_t>>;

using param_memlayout_rrr = dict_t<
    elem_v_t<tune_key::memory_layout_a, mem_layout::row_major>,
    elem_v_t<tune_key::memory_layout_b, mem_layout::row_major>,
    elem_v_t<tune_key::memory_layout_c, mem_layout::row_major>>;

using param_memspace_ggg = dict_t<
    elem_v_t<tune_key::memory_space_a, mem_space::global>,
    elem_v_t<tune_key::memory_space_b, mem_space::global>,
    elem_v_t<tune_key::memory_space_c, mem_space::global>>;

using param_performance_default = dict_t<
    elem_v_t<tune_key::wg_tile_k, 32UL, uint32_t>,
    elem_v_t<tune_key::prefetch_distance, 3UL, uint32_t>,
    elem_v_t<tune_key::periodic_sync_interval, 8UL, uint32_t>>;

template <gpu_arch arch_tag = gpu_arch::XeHpc>
using param_runtime_default = dict_t<
    elem_v_t<tune_key::pre_processing, tune_key_value::pre_processing_default>,
    elem_v_t<tune_key::mma_engine, mma_engine::xmx>,
    elem_v_t<tune_key::gpu_arch, arch_tag>,
    elem_t_t<
        tune_key::epilogue_policy,
        group::epilogue_policy_default<arch_tag>>,
    elem_v_t<
        tune_key::dispatch_policy,
        tune_key_value::dispatch_policy_default>,
    elem_t_t<
        tune_key::group_swizzle_policy,
        kernel::group_swizzle_default<arch_tag>>>;
} // namespace detail
template <gpu_arch arch_tag = gpu_arch::XeHpc>
using default_param_t = dict_t<>::template update_dict_t<
    detail::param_dtype_bf16_bf16_bf16>::
    template update_dict_t<detail::param_memlayout_rrr>::template update_dict_t<
        detail::param_memalignment_8_8_8>::
        template update_dict_t<detail::param_memspace_ggg>::
            template update_dict_t<detail::param_performance_default>::
                template update_dict_t<
                    detail::param_runtime_default<arch_tag>>::
                    template update_t<
                        elem_t_t<tune_key::data_type_acc, float>,
                        elem_v_t<
                            tune_key::global_kslicing_ratio,
                            1UL,
                            uint32_t>,
                        elem_v_t<tune_key::local_kslicing_ratio, 1UL, uint32_t>,
                        elem_t_t<tune_key::wg_tile_shape, shape<256, 256>>,
                        elem_t_t<tune_key::sg_tile_shape, shape<64, 32>>,
                        elem_v_t<
                            tune_key::param_optimizer_type,
                            tune_key_value::param_optimizer_dummy>,
                        elem_v_t<
                            tune_key::param_optimizer_level,
                            param_optimizer_level::full,
                            param_optimizer_level>>;

namespace kernel {
template <gpu_arch arch_tag = gpu_arch::XeHpc>
using param_kslicing_g1l1_t = default_param_t<arch_tag>::template update_t<
    elem_v_t<tune_key::global_kslicing_ratio, 1UL, uint32_t>,
    elem_v_t<tune_key::local_kslicing_ratio, 1UL, uint32_t>,
    elem_t_t<tune_key::wg_tile_shape, shape<256, 256>>,
    elem_v_t<tune_key::wg_tile_k, 32UL, uint32_t>,
    elem_t_t<tune_key::sg_tile_shape, shape<64, 32>>,
    elem_v_t<
        tune_key::dispatch_policy,
        tune_key_value::dispatch_policy_kslicing>>;

template <gpu_arch arch_tag = gpu_arch::XeHpc>
using param_kslicing_g2l1_t = default_param_t<arch_tag>::template update_t<
    elem_v_t<tune_key::global_kslicing_ratio, 2UL, uint32_t>,
    elem_v_t<tune_key::local_kslicing_ratio, 1UL, uint32_t>,
    elem_t_t<tune_key::wg_tile_shape, shape<256, 256>>,
    elem_v_t<tune_key::wg_tile_k, 32UL, uint32_t>,
    elem_t_t<tune_key::sg_tile_shape, shape<64, 32>>,
    elem_v_t<
        tune_key::dispatch_policy,
        tune_key_value::dispatch_policy_kslicing>>;

template <gpu_arch arch_tag = gpu_arch::XeHpc>
using param_kslicing_g1l2_t = default_param_t<arch_tag>::template update_t<
    elem_v_t<tune_key::global_kslicing_ratio, 1UL, uint32_t>,
    elem_v_t<tune_key::local_kslicing_ratio, 2UL, uint32_t>,
    elem_t_t<tune_key::wg_tile_shape, shape<128, 64>>,
    elem_v_t<tune_key::wg_tile_k, 32UL, uint32_t>,
    elem_t_t<tune_key::sg_tile_shape, shape<32, 16>>,
    elem_v_t<
        tune_key::dispatch_policy,
        tune_key_value::dispatch_policy_kslicing>>;

} // namespace kernel

namespace group {
template <gpu_arch arch_tag = gpu_arch::XeHpc>
using param_dict1_wg_t = default_param_t<arch_tag>::template update_t<
    elem_t_t<tune_key::data_type_acc, float>,
    elem_t_t<tune_key::wg_tile_shape, shape<256, 256>>,
    elem_v_t<tune_key::wg_tile_k, 32UL, uint32_t>,
    elem_t_t<tune_key::sg_tile_shape, shape<64, 32>>,
    elem_v_t<tune_key::prefetch_distance, 3UL, uint32_t>,
    elem_v_t<tune_key::periodic_sync_interval, 8UL, uint32_t>,
    elem_t_t<
        tune_key::epilogue_policy,
        group::epilogue_policy_default<arch_tag>>>;
}
} // namespace gpu::xetla
