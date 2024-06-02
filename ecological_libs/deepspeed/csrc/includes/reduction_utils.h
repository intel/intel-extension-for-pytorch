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

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.h>
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace reduce {

enum class ROpType {
    // Addition
    Add,

    // Maximum reduction
    Max,

    // Minimum reduction
    Min,
};

constexpr int max_threads = 1024;
constexpr int max_warps = max_threads / hw_warp_size;

/*
High level API. The API takes in a set of operations and variables
and performs that reduction operation on that variable. The reductions
of each of the arguments are completely independent of each other (
i.e., the val1-op1 combination has no impact on val2-op2).

Example usage:
``` cpp
float max_val;
float min_val;
reduce::block<rop::Max, rop::Min>(tb, warp, max_val, min_val);
```

TODO(cmikeh2): In theory, we might be able to do this sequentially with
device functions and rely on the assembler correctly behaving. My initial
instinct is this won't work, but if it does it would reduce implementation
cost significantly.

TODO(cmikeh2): We need to support sub-block reductions. The warp intrinsic
currently supports this (more incidentally than anything else). It is not
uncommon in something like softmax or a fused attention kernel to map multiple
reductions to a thread block, but each reduction itself is only scoped
to part of the threads (i.e block size = 512, 128 threads per reduction).
*/
template <ROpType Op, int warp_bound = max_warps>
DS_D_INLINE void block(sycl::group<3>& tb, sycl::sub_group& warp, float& val);

template <ROpType Op1, ROpType Op2, int warp_bound = max_warps>
DS_D_INLINE void block(sycl::group<3>& tb, sycl::sub_group& warp, float& val1, float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound = max_warps>
DS_D_INLINE void block(sycl::group<3>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound = max_warps>
DS_D_INLINE void block(sycl::group<3>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4);

/*
The partitioned block is a special case of the above where in the warps of a threadblock are
partitioned into separate independent reductions. For example, I might have an 8 warp thread block
in which each pair of warps is processing an independent piece of data. I would then reduce that
data with the something like the following:
``` cpp
float max_val;
reduce::partitioned_block<rop::Max, 2>(tb, warp, max_val);
```
After which, each pair of warps would have coherent data with each other. Note, this API will not
provide correct results if the number of warps per partition is not a power of 2.
*/
template <ROpType Op, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb, sycl::sub_group& warp, float& val);

template <ROpType Op1, ROpType Op2, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3,
                                   float& val4);

/*
Single element reduction primitives. Used inside serial collection
loops.

Example usage:
using rop = reduce::OpType;
float min = init<rop::Min>();
for (int i = 0; i < 4; i++) {
    min = reduce::element<rop::Min>(min, data[i]);
}
*/

template <ROpType Op, typename T>
DS_D_INLINE T element(const T lhs, const T rhs);

template <ROpType OType, typename T = float>
DS_D_INLINE T init();

/********************** Internal reduction APIs **********************/

/*
Single element "reductions". TODO(cmikeh2): this sort of "op" concept
should be refactored into its own implementation at some point. This interface
may be easily expanded for new types/operations, but the typical reductions
we need are covered with min/max/add on float.

NOTE: there is no mean reduction because that relies on knowledge of how
many values were already reduced into each scalar. Implementing this on top
of reduce should be straightforward (can just wrap the sum reduction) and
would be a good extension of the header.
*/

DS_D_INLINE int _warp_rank()
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    const int thread_rank =
        item_ct1.get_local_id(2) + item_ct1.get_local_id(1) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(0) * item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
    return thread_rank / hw_warp_size;
}

/* Float element reduce implementations */
template <>
DS_D_INLINE float element<ROpType::Add>(const float lhs, const float rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE float element<ROpType::Max>(const float lhs, const float rhs)
{
    return sycl::fmax((float)lhs, (float)rhs);
}

template <>
DS_D_INLINE float element<ROpType::Min>(const float lhs, const float rhs)
{
    return sycl::fmin((float)lhs, (float)rhs);
}

/* sycl::half element reduce implementation */
template <>
DS_D_INLINE sycl::half element<ROpType::Add>(const sycl::half lhs, const sycl::half rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE sycl::half element<ROpType::Max>(const sycl::half lhs, const sycl::half rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
DS_D_INLINE sycl::half element<ROpType::Min>(const sycl::half lhs, const sycl::half rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

/* sycl::half2 element reduce implementation */
template <>
DS_D_INLINE sycl::half2 element<ROpType::Add>(const sycl::half2 lhs, const sycl::half2 rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE sycl::half2 element<ROpType::Max>(const sycl::half2 lhs, const sycl::half2 rhs)
{
    sycl::half2 ret_val;
    ret_val.x() = (lhs.x() > rhs.x()) ? lhs.x() : rhs.x();
    ret_val.y() = (lhs.y() > rhs.y()) ? lhs.y() : rhs.y();
    return ret_val;
}

template <>
DS_D_INLINE sycl::half2 element<ROpType::Min>(const sycl::half2 lhs, const sycl::half2 rhs)
{
    sycl::half2 ret_val;
    ret_val.x() = (lhs.x() < rhs.x()) ? lhs.x() : rhs.x();
    ret_val.y() = (lhs.y() < rhs.y()) ? lhs.y() : rhs.y();
    return ret_val;
}

template <>
DS_D_INLINE int32_t element<ROpType::Add>(const int32_t lhs, const int32_t rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE int32_t element<ROpType::Max>(const int32_t lhs, const int32_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
DS_D_INLINE int32_t element<ROpType::Min>(const int32_t lhs, const int32_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template <>
DS_D_INLINE uint32_t element<ROpType::Add>(const uint32_t lhs, const uint32_t rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE uint32_t element<ROpType::Max>(const uint32_t lhs, const uint32_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
DS_D_INLINE uint32_t element<ROpType::Min>(const uint32_t lhs, const uint32_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template <>
DS_D_INLINE int64_t element<ROpType::Add>(const int64_t lhs, const int64_t rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE int64_t element<ROpType::Max>(const int64_t lhs, const int64_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
DS_D_INLINE int64_t element<ROpType::Min>(const int64_t lhs, const int64_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

/*
Reduction initialization primitives
*/
template <>
DS_D_INLINE float init<ROpType::Add>()
{
    return 0.0f;
}

template <>
DS_D_INLINE float init<ROpType::Min>()
{
    // Positive infinity
    return INFINITY;
}

template <>
DS_D_INLINE float init<ROpType::Max>()
{
    // Negative infinity
    return -INFINITY;
}

template <>
DS_D_INLINE sycl::half init<ROpType::Add>()
{
    return sycl::half(0.0);
}

template <>
DS_D_INLINE sycl::half init<ROpType::Min>()
{
    constexpr sycl::half inf = std::numeric_limits<sycl::half>::infinity();
    return sycl::half(inf);
}

template <>
DS_D_INLINE sycl::half init<ROpType::Max>()
{
    constexpr sycl::half neg_inf = -std::numeric_limits<sycl::half>::infinity();
    return sycl::half(neg_inf);
}

template <>
DS_D_INLINE sycl::half2 init<ROpType::Add>()
{
    return {0.0, 0.0};
}

template <>
DS_D_INLINE sycl::half2 init<ROpType::Min>()
{
    return {std::numeric_limits<sycl::half>::infinity(), std::numeric_limits<sycl::half>::infinity()};
}

template <>
DS_D_INLINE sycl::half2 init<ROpType::Max>()
{
    return {-std::numeric_limits<sycl::half>::infinity(), -std::numeric_limits<sycl::half>::infinity()};
}

template <>
DS_D_INLINE int32_t init<ROpType::Add>()
{
    return 0;
}

template <>
DS_D_INLINE int32_t init<ROpType::Min>()
{
    return 0x7FFFFFFF;
}

template <>
DS_D_INLINE int32_t init<ROpType::Max>()
{
    return 0x80000000;
}

template <>
DS_D_INLINE uint32_t init<ROpType::Add>()
{
    return 0;
}

template <>
DS_D_INLINE uint32_t init<ROpType::Min>()
{
    return 0xFFFFFFFF;
}

template <>
DS_D_INLINE uint32_t init<ROpType::Max>()
{
    return 0;
}

template <>
DS_D_INLINE int64_t init<ROpType::Add>()
{
    return 0;
}

template <>
DS_D_INLINE int64_t init<ROpType::Min>()
{
    return 0x7FFFFFFFFFFFFFFF;
}

template <>
DS_D_INLINE int64_t init<ROpType::Max>()
{
    return 0x8000000000000000;
}

template <>
DS_D_INLINE uint64_t init<ROpType::Add>()
{
    return 0;
}

template <>
DS_D_INLINE uint64_t init<ROpType::Min>()
{
    return 0xFFFFFFFFFFFFFFFF;
}

template <>
DS_D_INLINE uint64_t init<ROpType::Max>()
{
    return 0;
}

template <ROpType Op, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op, T>();
}

template <ROpType Op1, ROpType Op2, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
}

template <ROpType Op1, ROpType Op2, ROpType Op3, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
    data[2] = init<Op3, T>();
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
    data[2] = init<Op3, T>();
    data[3] = init<Op4, T>();
}

/*
Warp reduction primitives

`reduction_width` is an unsafe template parameter, that is that
when using `reduction_width` < hw_warp_size the warp is partitioned
into `hw_warp_size` / `reduction_width` groups of partial sums.

If someone can figure out how to use variadic templates in a reasonable way
here (fold is C++17 only and I don't think helps and recursion feels like
huge overkill that harms readability) that would be wonderful.
*/

template <typename T, ROpType Op, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(sycl::sub_group& warp, T* data)
{
    auto tb = sycl::ext::oneapi::experimental::this_group<3>();
    auto reduce_width_ = tb.get_local_range(2) < reduce_width ? tb.get_local_range(2) : reduce_width;
#pragma unroll
    for (int i = 1; i < reduce_width_; i *= 2) {
        data[0] = element<Op>(data[0],
                              sycl::permute_group_by_xor(
                                  sycl::ext::oneapi::experimental::this_sub_group(), data[0], i));
    }
}

template <typename T, ROpType Op1, ROpType Op2, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(sycl::sub_group& warp, T* data)
{
    auto tb = sycl::ext::oneapi::experimental::this_group<3>();
    auto reduce_width_ = tb.get_local_range(2) < reduce_width ? tb.get_local_range(2) : reduce_width;
#pragma unroll
    for (int i = 1; i < reduce_width_; i *= 2) {
        data[0] = element<Op1>(data[0],
                               sycl::permute_group_by_xor(
                                   sycl::ext::oneapi::experimental::this_sub_group(), data[0], i));
        data[1] = element<Op2>(data[1],
                               sycl::permute_group_by_xor(
                                   sycl::ext::oneapi::experimental::this_sub_group(), data[1], i));
    }
}

template <typename T, ROpType Op1, ROpType Op2, ROpType Op3, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(sycl::sub_group& warp, T* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shuffle_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shuffle_xor(data[1], i));
        data[2] = element<Op3>(data[2], warp.shuffle_xor(data[2], i));
    }
}

template <typename T,
          ROpType Op1,
          ROpType Op2,
          ROpType Op3,
          ROpType Op4,
          int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(sycl::sub_group& warp, T* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shuffle_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shuffle_xor(data[1], i));
        data[2] = element<Op3>(data[2], warp.shuffle_xor(data[2], i));
        data[3] = element<Op4>(data[3], warp.shuffle_xor(data[3], i));
    }
}

/*
Implementation for primary block reduction that serves both `block` and
`partitioned_block`.

Total warps refers to the reduction width of the reduction, not
the number of warps in the block (which may exceed that
if the block is partitioned or if we do a conservative bound at
compile time).
*/
template <typename T, int total_warps, ROpType... Ops>
DS_D_INLINE void _block(sycl::group<3>& tb, sycl::sub_group& warp_arg, T* data)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int elems = sizeof...(Ops);
    constexpr int bytes = sizeof(T);
    // Unused when `partition_size == 1` or total_warps == 1
    /*
    DPCT1115:0: The sycl::ext::oneapi::group_local_memory is used to allocate group-local memory at
    the none kernel functor scope of a work-group data parallel kernel. You may need to adjust the
    code.
    */
    auto& reduce_buffer = *sycl::ext::oneapi::group_local_memory_for_overwrite<T[max_warps * elems]>(
        sycl::ext::oneapi::experimental::this_group<3>());

    /*
    DPCT1007:7: Migration of cooperative_groups::thread_block_tile::meta_group_size is not
    supported.
    */
    const int running_warps = warp_arg.get_group_range().size();

    // Always perform warp-scope reduction
    _warp<T, Ops...>(warp_arg, data);

    // If max_warps == 1 let's skip the runtime check
    if (total_warps != 1) {
        if (sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id() == 0) {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(reduce_buffer + elems * _warp_rank() + i, data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if
        there is no access to global memory.
        */
        item_ct1.barrier();

        if (_warp_rank() == 0) {
            if (sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id() <
                running_warps) {
#pragma unroll
                for (int i = 0; i < elems; i++) {
                    mem_access::load_shared<bytes>(
                        data + i,
                        reduce_buffer +
                            elems * sycl::ext::oneapi::experimental::this_sub_group()
                                        .get_local_linear_id() +
                            i);
                }
            } else {
                init<Ops...>(data);
            }

            _warp<T, Ops..., total_warps>(warp_arg, data);

#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(
                    reduce_buffer +
                        elems * sycl::ext::oneapi::experimental::this_sub_group()
                                    .get_local_linear_id() +
                        i,
                    data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if
        there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for (int i = 0; i < elems; i++) {
            mem_access::load_shared<bytes>(data + i, reduce_buffer + _warp_rank() * elems + i);
        }
    }
}

/*
Main API implementations. For the most part, they just convert the individual
variables into arrays, which makes working with them easier with a single
implementation. In theory, we could use the `_block` implementation as another
option, but the nature of using a pointer is a little less safe and this allows
us to obfuscate the details of the partitioned implementation.
*/
template <ROpType Op, int warp_bound>
DS_D_INLINE void block(sycl::group<3>& tb, sycl::sub_group& warp, float& val)
{
    _block<float, warp_bound, Op>(tb, warp, &val);
}

template <ROpType Op1, ROpType Op2, int warp_bound>
DS_D_INLINE void block(sycl::group<3>& tb, sycl::sub_group& warp, float& val1, float& val2)
{
    float data[2] = {val1, val2};
    _block<float, warp_bound, Op1, Op2>(tb, warp, data);
    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound>
DS_D_INLINE void block(sycl::group<3>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3)
{
    float data[3] = {val1, val2, val3};
    _block<float, warp_bound, Op1, Op2, Op3>(tb, warp, data);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound>
DS_D_INLINE void block(sycl::group<3>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4)
{
    float data[4] = {val1, val2, val3, val4};
    _block<float, warp_bound, Op1, Op2, Op3, Op4>(tb, warp, data);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

/*
Note: for the partitioned blocks, the implementation does not support non-power of 2 blocks in order
to shorten block scale reduction length.
*/
template <ROpType Op, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb, sycl::sub_group& warp, float& val)
{
    if (num_threads <= hw_warp_size) {
        _warp<float, Op, num_threads>(warp, &val);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        _block<float, num_warps, Op>(tb, warp, &val);
    }
}

template <ROpType Op1, ROpType Op2, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2)
{
    float data[2] = {val1, val2};

    if (num_threads <= hw_warp_size) {
        _warp<float, Op1, Op2, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        _block<float, num_warps, Op1, Op2>(tb, warp, data);
    }

    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3)
{
    float data[3] = {val1, val2, val3};

    if (num_threads <= hw_warp_size) {
        _warp<float, Op1, Op2, Op3, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        _block<float, num_warps, Op1, Op2, Op3>(tb, warp, data);
    }

    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int num_threads>
DS_D_INLINE void partitioned_block(sycl::group<3>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3,
                                   float& val4)
{
    float data[4] = {val1, val2, val3, val4};

    if (num_threads <= hw_warp_size) {
        _warp<float, Op1, Op2, Op3, Op4, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        _block<float, num_warps, Op1, Op2, Op3, Op4>(tb, warp, data);
    }

    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

/*
Arg-reduce is a specialization of the above. We only support this with a single reduction
parameter. This only works for max/min reductions.
*/

__dpct_align__(8) struct IdxReduceResult {
    /*
    NOTE: ORDERING MATTERS HERE! The idx is the least significant set of bits
    and the val is the most significant. Changing the order of this declaration
    will break the code.
    */
    int idx;
    float val;
};

template <ROpType Op, int warpBound>
DS_D_INLINE IdxReduceResult
idx_reduce(sycl::group<3>& tb, sycl::sub_group& warp, float val, int idx)
{
    IdxReduceResult res = {idx, val};

    // Clear out the nan. This shouldn't be an issue for our initial applications
    if (sycl::isnan(val)) res.val = init<Op>();

    // Can do float compares as integers. By packing the index into the lower bits
    // we can just do a single int64 rather than a branch, compare, and select.
    // One side benefit of this is that it is by nature a stable algorithm and
    // will always bias ties to the higher index.
    int64_t* res_as_int = reinterpret_cast<int64_t*>(&res);

    // The way floating point compare works is normally to perform a sign comparison
    // and if they match, then do a comparison of the rest of the bits as unsigned
    // integers. Since we are bundling these, that means for negative values we need
    // to reverse the sort order, which we can do with an XOR.
    if (val < 0) { *res_as_int ^= 0x7fffffff00000000; }

    _block<int64_t, warpBound, Op>(tb, warp, res_as_int);

    // Sign bit is preserved, so we can check if we need to invert the mantissa back
    if (res.val < 0) { *res_as_int ^= 0x7fffffff00000000; }

    return res;
}

}  // namespace reduce
