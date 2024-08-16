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

#include <experimental/kernel/int4_dequantize/config.hpp>

namespace gpu::xetla::kernel {

/// @brief
///
/// @tparam dtype_qweight_  qweight data type.
/// @tparam dtype_scale_ scale data type.
/// @tparam dtype_zp_ zero point data
/// @tparam dtype_dequant_weight_  dequant_weight data type.
/// @tparam mem_layout_dequant_weight_ dequant_weight memory layout.
/// @tparam quant_info quant_mode, blocksize, qweight_layout info.
/// @tparam int4_dequantize_attr_ parallel-related attribute.
/// @tparam arch_ HW architecture.
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
    gpu_arch arch_,
    typename enable = void>
struct int4_dequantize_t {};

} // namespace gpu::xetla::kernel
