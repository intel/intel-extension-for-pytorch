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
#include <cassert>
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "reduction_utils.h"

#pragma once

using rop = reduce::ROpType;

namespace quantize {
constexpr int granularity = 16;
constexpr int h_per_load = granularity / sizeof(sycl::half);
constexpr int h2_per_load = granularity / sizeof(sycl::half2);
constexpr int max_threads = 1024;

/*
Class to hold the quantization parameters for a given tensor.
Holds the implementation of the quantization operation.
*/
template <Type qType, int numBits>
class Params {
public:
    /*
    Quantization implementation, supports
    1) 4 Bit
    2) 8 Bit
    3) Symmetric
    4) Asymmetric
    Function Arguments :
        val : The sycl::half value to quantize.
    */
    DS_D_INLINE int8_t quantize(sycl::half val);

    template <typename T>
    DS_D_INLINE T dequantize(int8_t val);

    DS_D_INLINE void store(float* params, int group_index);

    // Initialize from memory
    DS_D_INLINE Params(const float* params, int group_index);
};

template <int numBits>
class Params<Type::Symmetric, numBits> {
public:
    float scale;

    DS_D_INLINE Params(float max)
    {
        if (max == 0) {
            scale = 1.0;
        } else {
            scale = (1 << numBits) / (2 * max);
        }
    }

    DS_D_INLINE int8_t quantize(sycl::half val)
    {
        constexpr int32_t q_min = -(1 << (numBits - 1));
        constexpr int32_t q_max = (1 << (numBits - 1)) - 1;

        float val_f = conversion::to<float>(val) * scale;
        int32_t data_i32 = conversion::to<int32_t>(val_f);
        data_i32 = dpct::min(sycl::max(data_i32, q_min), q_max);
        return (int8_t)data_i32;
    }

    template <typename T>
    DS_D_INLINE T dequantize(int8_t val)
    {
        const float val_deq_f = conversion::to<float>(val) * scale;
        return conversion::to<T>(val_deq_f);
    }

    DS_D_INLINE void store(float* params, int group_index)
    {
        const float store_scale = 1 / scale;
        mem_access::store_global<sizeof(float)>(params + group_index, &store_scale);
    }

    DS_D_INLINE Params(const float* params, int group_index)
    {
        mem_access::load_global<sizeof(float)>(&scale, params + group_index);
    }
};

template <int numBits>
class Params<Type::Asymmetric, numBits> {
public:
    float scale;
    float offset;

    DS_D_INLINE Params(float max, float min)
    {
        if (max == min) {
            scale = 1.0;
        } else {
            scale = ((1 << numBits)) / (max - min);
        }
        offset = (max + min) / 2;
    }

    DS_D_INLINE int8_t quantize(sycl::half val)
    {
        constexpr int32_t q_min = -(1 << (numBits - 1));
        constexpr int32_t q_max = (1 << (numBits - 1)) - 1;

        float val_f = (conversion::to<float>(val) - offset) * scale;
        int32_t data_i32 = conversion::to<int32_t>(val_f);
        data_i32 = dpct::min(sycl::max(data_i32, q_min), q_max);
        return (int8_t)data_i32;
    }

    template <typename T>
    DS_D_INLINE T dequantize(int8_t val)
    {
        const float val_deq_f = ((conversion::to<float>(val)) * scale) + offset;
        return conversion::to<sycl::half>(val_deq_f);
    }

    DS_D_INLINE void store(float* params, int group_index)
    {
        // Codegen should turn this into stg.64
        const float store_scale = 1 / scale;
        mem_access::store_global<sizeof(float)>(params + 2 * group_index, &store_scale);
        mem_access::store_global<sizeof(float)>(params + 2 * group_index + 1, &offset);
    }

    DS_D_INLINE Params(const float* params, int group_index)
    {
        // Codegen should turn this into ldg.64
        mem_access::load_global<sizeof(float)>(&scale, params + 2 * group_index);
        mem_access::load_global<sizeof(float)>(&offset, params + 2 * group_index + 1);
    }
};

/*
Group stats tracks the necessary statistics about the quantized group
to abstract the particulars for the main loop.
*/
template <Type qType>
class GroupStats {
public:
    DS_D_INLINE void update(sycl::half2 val);

    DS_D_INLINE void reduce(sycl::group<3>& tb, sycl::sub_group& warp);
};

template <>
class GroupStats<Type::Symmetric> {
public:
    // Symmetric quantization only tracks the maximum absolute value
    sycl::half2 cur_max;
    float max;

    /*
    Technically, this would give bad results if there
    are 0 values to process since the reduction would
    give -inf instead of 0. We do not consider this
    to be a reasonable edge case.
    */
    DS_D_INLINE GroupStats() { cur_max = reduce::init<rop::Max, sycl::half2>(); }

    /*
    Updated the running absmax used to calculate params.
    Function Arguments :
        val : The sycl::half2 value to update the running min and max with.
    */
    DS_D_INLINE void update(sycl::half2 val)
    {
        cur_max = reduce::element<rop::Max>(cur_max, sycl::fabs(val));
    }

    /*
    Function to return calculated quantization params.
    Template Arguments :
        numBits -   Number of bits in quantized element.    int : 8 or 4
    Function Arguments :
        tb      -   Threadblock object. auto
        warp    -   Warp object.        auto
    */
    template <int numBits, int threads_per_group>
    DS_D_INLINE Params<Type::Symmetric, numBits> get_params(sycl::group<3>& tb,
                                                            sycl::sub_group& warp)
    {
        const sycl::float2 partial_max = conversion::to<sycl::float2>(cur_max);
        float max = reduce::element<rop::Max>(partial_max.x(), partial_max.y());

        reduce::partitioned_block<rop::Max, threads_per_group>(tb, warp, max);
        Params<Type::Symmetric, numBits> params(max);

        return params;
    }
};

template <>
class GroupStats<Type::Asymmetric> {
public:
    sycl::half2 cur_max;
    sycl::half2 cur_min;

    /*
    Initialize cur_max to -inf, cur_min to inf since
    we are doing a true range analysis.
    */
    DS_D_INLINE GroupStats()
    {
        cur_max = reduce::init<rop::Max, sycl::half2>();
        cur_min = reduce::init<rop::Min, sycl::half2>();
    }

    /*
    Updated the running min and max used to calculate params.
    Function Arguments :
        val : The sycl::half2 value to update the running min and max with.
    */
    DS_D_INLINE void update(sycl::half2 val)
    {
        cur_max = reduce::element<rop::Max>(cur_max, val);
        cur_min = reduce::element<rop::Min>(cur_min, val);
    }

    /*
    Function to return calculated quantization params.
    Template Arguments :
        numBits -   Number of bits in quantized element.    int : 8 or 4
    Function Arguments :
        tb      -   Threadblock object. auto
        warp    -   Warp object.        auto
    */
    template <int numBits, int threads_per_group>
    DS_D_INLINE Params<Type::Asymmetric, numBits> get_params(sycl::group<3>& tb,
                                                             sycl::sub_group& warp)
    {
        const sycl::float2 partial_max = conversion::to<sycl::float2>(cur_max);
        float max = reduce::element<rop::Max>(partial_max.x(), partial_max.y());

        const sycl::float2 partial_min = conversion::to<sycl::float2>(cur_min);
        float min = reduce::element<rop::Min>(partial_min.x(), partial_min.y());

        reduce::partitioned_block<rop::Max, rop::Min, threads_per_group>(tb, warp, max, min);

        Params<Type::Asymmetric, numBits> params(max, min);

        return params;
    }
};

/*
Device function that quantizes 16 bytes of sycl::half type input data.
Template Arguments :
    numBits -   Number of bits in quantized element.    int : 8 or 4
    qType   - Type of quantization to perform.          Type::Symmetric or Type::Asymmetric
Function Arguments :
    local_output -  Pointer to local memory to store quantized data.    int8_t*
    data         -  Pointer to input data.                              sycl::half*
    Params       -  Parameters for quantization.                        Params<qType, numBits>
*/
template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output,
                        const sycl::half* data,
                        Params<qType, numBits> q_params);

/*
Device function that quantizes 16 bytes of sycl::half2 type input data.
Template Arguments :
    numBits -   Number of bits in quantized element.    int : 8 or 4
    qType   -   Type of quantization to perform.        Type::Symmetric or Type::Asymmetric
Function Arguments :
    local_output -  Pointer to local memory to store quantized data.    int8_t*
    data         -  Pointer to input data.                              sycl::half2*
    Params       -  Parameters for quantization.                        Params<qType, numBits>
*/
template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output,
                        const sycl::half2* data,
                        Params<qType, numBits> q_params);

/*
Helper function to do serial reduction on register-file arrays.
Template Arguments :
    qType       -   Type of quantization to perform.        Type::Symmetric or Type::Asymmetric
    numChunks   -   Number of bits in quantized element.    int : 8 or 4
Function Arguments :
    local_buffer    -   Pointer memory with input half2 data to be quantized.
*/
template <Type qType, int numChunks>
DS_D_INLINE GroupStats<qType> _local_serial_reduce(sycl::half2* local_buffer);

/*
The main loop of the kernel that quantizes array in local memory of sycl::half2 type input data, when
Quantization parameters are pre-computed.
Template Arguments :
    qType       -   Type of quantization to perform.            Type::Symmetric or Type::Asymmetric
    numBits     -   Number of bits in quantized element.        int : 8 or 4
    numChunks   -   Number of chunks(16 bytes of Input data).   int : 8 or 4
Function Arguments :
    local_buffer    -   Pointer memory with input half2 data to be quantized.
    scales          -   Pointer to output scales.
    offsets         -   Pointer to output offsets.
    output_data     -   Pointer to output data.
    elems_per_group -   Number of elements to quantize in a group.
    q_params        -   Quantization parameters.
*/
template <int numBits, Type qType, int numChunks, int threads_per_group, int max_threads>
DS_D_INLINE void local_array(sycl::group<3>& tb,
                             sycl::sub_group& warp,
                             sycl::half2* local_buffer,
                             float* __restrict__ scales,
                             float* __restrict__ offsets,
                             int8_t* __restrict__ output_data,
                             const int& elems_per_group,
                             const int& groups,
                             Params<qType, numBits> q_params);

/*
The main loop of the kernel that quantizes array in local memory of sycl::half2 type input data.
This function computes quantization parameters for each group.
Template Arguments :
    qType   -   Type of quantization to perform.                Type::Symmetric or Type::Asymmetric
    numBits     -   Number of bits in quantized element.        int : 8 or 4
    numChunks   -   Number of chunks(16 bytes of Input data).   int : 8 or 4
Function Arguments :
    local_buffer    -   Pointer memory with input half2 data to be quantized.
    scales          -   Pointer to output scales.
    offsets         -   Pointer to output offsets.
    output_data     -   Pointer to output data.
    elems_per_group -   Number of elements to quantize in a group.
*/
template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
void local_array(sycl::half2* local_buffer,
                 float* __restrict__ scales,
                 float* __restrict__ offsets,
                 int8_t* __restrict__ output_data,
                 const int& elems_per_group,
                 const int& groups);

template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output,
                        const sycl::half* data,
                        Params<qType, numBits> q_params)
{
    constexpr int32_t elems = 16 / sizeof(sycl::half);
    constexpr int32_t num_elems_packed = 8 / numBits;

#pragma unroll
    for (int i = 0, oi = 0; i < elems; i += num_elems_packed, oi++) {
        if (num_elems_packed == 1) {
            // TODO(cmikeh2): refactor to use conversion utils
            local_output[i] = q_params.quantize(data[i]);
        } else if (num_elems_packed == 2) {
            int8_t data_i8_1 = q_params.quantize(data[i]);
            int8_t data_i8_2 = q_params.quantize(data[i + 1]);
            auto data_i8 = PackedInt4{data_i8_2, data_i8_1};
            local_output[oi] = *((int8_t*)(&data_i8));
        }
    }
}

template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output,
                        const sycl::half2* data,
                        Params<qType, numBits> q_params)
{
    const sycl::half* data_cast = reinterpret_cast<const sycl::half*>(data);
    _chunk<numBits>(local_output, data_cast, q_params);
}

template <Type qType, int numChunks>
DS_D_INLINE GroupStats<qType> _local_serial_reduce(sycl::half2* local_buffer)
{
    GroupStats<qType> stats;
#pragma unroll
    for (int i = 0; i < numChunks * h2_per_load; i++) { stats.update(local_buffer[i]); }

    return stats;
}

template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
DS_D_INLINE void local_array(sycl::group<3>& tb,
                             sycl::sub_group& warp,
                             sycl::half2* local_buffer,
                             float* __restrict__ global_params,
                             int8_t* __restrict__ output_data,
                             const int& elems_per_group,
                             const int& groups,
                             Params<qType, numBits> q_params)
{
    constexpr int num_ele_int8 = 8 / numBits;
    constexpr int num_int8_out = quantize::h_per_load / num_ele_int8;

    // Indexing offsets
    const int block_num =
        (tb.get_group_id()[2] * max_threads / threads_per_group) + tb.get_local_id()[1];
    const int block_offset = block_num * elems_per_group;
    const int elem_offset = tb.get_local_id()[2] * quantize::h_per_load;
    const int base_offset = (block_offset + elem_offset) / num_ele_int8;
    const int stride = sycl::ext::oneapi::experimental::this_group<3>().get_local_linear_range() *
                       quantize::h_per_load / num_ele_int8;

    int8_t local_output[num_int8_out];

    if (tb.get_local_id()[2] == 0 && block_num < groups) {
        q_params.store(
            global_params,
            (tb.get_group_id()[2] * max_threads / threads_per_group) + tb.get_local_id()[1]);
    }
#pragma unroll
    for (int i = 0; i < numChunks; i++) {
        if (elem_offset + i * stride * num_ele_int8 < elems_per_group && block_num < groups) {
            quantize::_chunk<numBits, qType>(
                local_output, local_buffer + i * quantize::h2_per_load, q_params);
            mem_access::store_global<num_int8_out>(output_data + (base_offset + i * stride),
                                                   local_output);
        }
    }
}

template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
DS_D_INLINE void local_array(sycl::group<3>& tb,
                             sycl::sub_group& warp,
                             sycl::half* local_buffer,
                             float* __restrict__ global_params,
                             int8_t* __restrict__ output_data,
                             const int& elems_per_group,
                             const int& groups,
                             Params<qType, numBits> q_params)
{
    sycl::half2* local_buffer_h2 = reinterpret_cast<sycl::half2*>(local_buffer);

    quantize::local_array<qType, numBits, numChunks, threads_per_group, max_threads>(
        tb, warp, local_buffer, global_params, output_data, elems_per_group, groups, q_params);
}

template <Type qType,
          int numBits,
          int numChunks,
          int threads_per_group = max_threads,
          int max_threads = 256>
void local_array(sycl::half2* local_buffer,
                 float* __restrict__ global_params,
                 int8_t* __restrict__ output_data,
                 const int& elems_per_group,
                 const int& groups)
{
    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    auto group_stats = _local_serial_reduce<qType, numChunks>(local_buffer);
    auto params = group_stats.template get_params<numBits, threads_per_group>(tb, warp);

    quantize::local_array<qType, numBits, numChunks, threads_per_group, max_threads>(
        tb, warp, local_buffer, global_params, output_data, elems_per_group, groups, params);
}

template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
void local_array(sycl::half* local_buffer,
                 float* __restrict__ global_params,
                 int8_t* __restrict__ output_data,
                 const int& elems_per_group,
                 const int& groups)
{
    sycl::half2* local_buffer_h2 = reinterpret_cast<sycl::half2*>(local_buffer);
    quantize::local_array<qType, numBits, numChunks, threads_per_group, max_threads>(
        local_buffer_h2, global_params, output_data, elems_per_group, groups);
}

}  // namespace quantize
