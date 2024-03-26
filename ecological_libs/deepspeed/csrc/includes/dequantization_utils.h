/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
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
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.h>
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "quantization.h"
#include "quantization_utils.h"

#pragma once

namespace dequantize {
using Type = quantize::Type;

template <Type qType, int numBits>
using Params = quantize::Params<qType, numBits>;

constexpr int granularity = quantize::granularity;
using PackedInt4 = quantize::PackedInt4;

constexpr int h_per_chunk = granularity / sizeof(sycl::half);
constexpr int h2_per_chunk = granularity / sizeof(sycl::half2);

/*
Device function that reads quantized data from global memory, dequantizes
it, and stores it to global memory.
Template Arguments :
    numBits - Number of bits in quantized element.      int: 4, 8
    qType - Type of quantization to perform.            Type::Symmetric or Type::Asymmetric
    unroll - Number of load steps to internally unroll  int
    threads - Number of threads to perform dequant      int
Function arguments:
    global_output - sycl::half pointer in global memory
    data - Quantized data in global memory
    global_params - Quantization parameters in global memory
    elems_per_group - Number of elements in each quantization group
    total_elems - Tensor size (note, does not need to be multiple of elems_per_group)
*/
template <int numBits, Type qType, int unroll, int threads>
DS_D_INLINE void to_global(sycl::half* global_output,
                           const int8_t* data,
                           const float* global_params,
                           const int elems_per_group,
                           const int total_elems);

/*
Device function that quantizes 16 bytes of sycl::half type input data.
Template Arguments :
    numBits -   Number of bits in quantized element.    int : 8 or 4
    qType   - Type of quantization to perform.          Type::Symmetric or Type::Asymmetric
Function Arguments :
    local_output -  Local array to store dequantized data       sycl::half* or sycl::half2*
    data         -  Pointer to quantized input data.            int8_t*
    Params       -  Parameters for quantization.                Params<qType, numBits>
*/
template <int numBits, Type qType>
DS_D_INLINE void chunk(sycl::half2* local_output,
                       const int8_t* data,
                       Params<qType, numBits> q_params);

template <typename T, int numBits, Type qType>
DS_D_INLINE void chunk(T* local_output, const int8_t* data, Params<qType, numBits> q_params);

/**************** Implementations ******************/

template <typename T, int numBits, Type qType>
DS_D_INLINE void chunk(T* local_output, const int8_t* data, Params<qType, numBits> q_params)
{
    constexpr int32_t num_elems_packed = 8 / numBits;
    constexpr int32_t iters = h_per_chunk / num_elems_packed;

#pragma unroll
    for (int i = 0; i < iters; i++) {
        if constexpr (num_elems_packed == 1) {
            local_output[i] = q_params.template dequantize<T>(data[i]);
        } else {
            auto accessible_data = *(PackedInt4*)(&data[i]);
            local_output[2 * i] = q_params.template dequantize<T>(accessible_data.low);
            local_output[2 * i + 1] = q_params.template dequantize<T>(accessible_data.high);
        }
    }
}

template <int numBits, Type qType>
DS_D_INLINE void chunk(sycl::half2* local_output,
                       const int8_t* data,
                       Params<qType, numBits> q_params)
{
    sycl::half* local_output_cast = reinterpret_cast<sycl::half*>(local_output);
    chunk<sycl::half, numBits>(local_output_cast, data, q_params);
}

template <typename T, int numBits, Type qType, int unroll, int threads>
/*
DPCT1110:46: The total declared local variable size in device function _to_global exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
DS_D_INLINE void _to_global(T* global_output,
                            const int8_t* data,
                            const float* global_params,
                            const int elems_per_group,
                            const int total_elems)
{
    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    // Load constants
    // TODO(cmikeh2): Refactor into functions?
    constexpr int load_granularity = (granularity / (sizeof(T))) / (numBits == 8 ? 1 : 2);
    constexpr int load_step_stride = load_granularity * threads;
    constexpr int load_block_stride = load_step_stride * unroll;

    // Store constants
    constexpr int T_per_chunk = granularity / sizeof(T);
    constexpr int store_step_stride = T_per_chunk * threads;
    constexpr int store_block_stride = store_step_stride * unroll;

    // Load offsets
    const int load_block_offset = tb.get_group_id()[2] * load_block_stride;
    // Note: we can use `load_granularity` since the dtype is `int8_t`.
    const int load_thread_offset = tb.get_local_id()[2] * load_granularity;
    const int8_t* load_base = data + load_block_offset + load_thread_offset;

    // Store offsets
    const int store_block_offset = tb.get_group_id()[2] * store_block_stride;
    const int store_thread_offset = tb.get_local_id()[2] * T_per_chunk;
    const int elem_id_base = store_block_offset + store_thread_offset;

    int8_t local_load_buffer[load_granularity * unroll];
    T local_dequant_buffer[T_per_chunk * unroll];

    /*
    Note: Splitting this loop in half gave about 3-5% performance increase for reasons that aren't
    totally clear to me, so this is a deliberately weird code structure.
    */
#pragma unroll
    for (int i = 0; i < unroll; i++) {
        const int elem_id_iter = elem_id_base + i * store_step_stride;

        if (elem_id_iter < total_elems) {
            mem_access::load_global<load_granularity>(local_load_buffer + i * load_granularity,
                                                      load_base + i * load_step_stride);
        }
    }

#pragma unroll
    for (int i = 0; i < unroll; i++) {
        const int elem_id_iter = elem_id_base + i * store_step_stride;
        if (elem_id_iter < total_elems) {
            // TODO(cmikeh2): Can we amortize this division? Perform once on the first iteration and
            // use indexing math to do division free interpolation of the successive groups?
            const int group_index = elem_id_iter / elems_per_group;
            Params<qType, numBits> q_params(global_params, group_index);

            chunk<T, numBits, qType>(local_dequant_buffer + i * T_per_chunk,
                                     local_load_buffer + i * load_granularity,
                                     q_params);
            mem_access::store_global<granularity>(global_output + elem_id_iter,
                                                  local_dequant_buffer + i * T_per_chunk);
        }
    }
}

template <typename T, int numBits, Type qType, int unroll, int threads>
DS_D_INLINE void to_global(T* global_output,
                           const int8_t* data,
                           const float* global_params,
                           const int elems_per_group,
                           const int total_elems)
{
    if constexpr (numBits == 4 || numBits == 8) {
        _to_global<T, numBits, qType, unroll, threads>(
            global_output, data, global_params, elems_per_group, total_elems);
    } else if constexpr (numBits == 3) {
        // TODO(cmikeh2): Need this implementation
        assert(false);
    } else {
        assert(false);
    }
}

}  // namespace dequantize
