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
#include "ds_kernel_utils.h"

/////////////////////////////// Memory Access Utils ///////////////////////////////
namespace mem_access {

enum class LoadPolicy {
    CacheAll,       // Cache at all levels
    CacheGlobal,    // Cache at L2 only
    CacheStreaming  // Cache with evict first policy
};

enum class StorePolicy {
    Writeback,      // Cache in L1, write-back on eviction
    CacheGlobal,    // Bypass L1, write-back on eviction
    CacheStreaming  // Allocate cache line with evict first policy
};

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
__dpct_inline__ void load_global(void* dst, const void* src);

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
__dpct_inline__ void load_global(void* dst, const void* src, bool do_access);

// Shared accesses have no cache policy
template <int AccessSize>
__dpct_inline__ void load_shared(void* dst, const void* src);

template <int AccessSize>
__dpct_inline__ void load_shared(void* dst, const void* src, bool do_access);

template <int AccessSize, StorePolicy policy = StorePolicy::Writeback>
__dpct_inline__ void store_global(void* dst, const void* src);

// Shared accesses have no cache policy
template <int AccessSize>
__dpct_inline__ void store_shared(void* dst, const void* src);

#ifdef ASYNC_COPY_AVAILABLE
template <int AccessSize>
__device__ __forceinline__ void memcpy_async(void* shr, const void* gbl);

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_nop(void* shr, const void* gbl, bool predicate);

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_zero(void* shr, const void* gbl, bool predicate);

__device__ __forceinline__ void memcpy_async_fence();

template <int stages>
__device__ __forceinline__ void memcpy_async_wait();

template <int stages>
__device__ __forceinline__ void tail_complete_wait(int remaining_stages);
#endif

// Util for tracking pipeline buffers
// TODO: Evaluate whether this should also be guarded by ASYNC_COPY_AVAILABLE
template <int max>
class BufferTracker {
public:
    int current_state;

    __dpct_inline__ BufferTracker() : current_state(0) {}

    __dpct_inline__ int get()
    {
        int return_val = current_state++;
        current_state = (current_state == max ? 0 : current_state);
        return return_val;
    }
};

__dpct_inline__ uint32_t lane_id()
{
#ifdef PTX_AVAILABLE
    unsigned int lane_id;
    lane_id =
        sycl::ext::oneapi::experimental::this_nd_item<3>().get_sub_group().get_local_linear_id();
    return lane_id;
#else
    return sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_id(2) &
           (sycl::ext::oneapi::experimental::this_sub_group().get_local_range().get(0) -
            1);  // Portable
#endif
}

/////////// Load Global ///////////
template <>
__dpct_inline__ void load_global<16>(void* dst, const void* src)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:1: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
                 : "l"(src));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<16>(void* dst, const void* src, bool do_access)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:2: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
        : "l"(src), "r"((int)do_access));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<16, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:3: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
                 : "l"(src));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<16, LoadPolicy::CacheGlobal>(void* dst,
                                                              const void* src,
                                                              bool do_access)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:4: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
        : "l"(src), "r"((int)do_access));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<16, LoadPolicy::CacheStreaming>(void* dst, const void* src)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:5: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
                 : "l"(src));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<16, LoadPolicy::CacheStreaming>(void* dst,
                                                                 const void* src,
                                                                 bool do_access)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:6: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
        : "l"(src), "r"((int)do_access));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<8>(void* dst, const void* src)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:7: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y())
                 : "l"(src));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<8>(void* dst, const void* src, bool do_access)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:8: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.global.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y())
        : "l"(src), "r"((int)do_access));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<8, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:9: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y())
                 : "l"(src));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<8, LoadPolicy::CacheGlobal>(void* dst,
                                                             const void* src,
                                                             bool do_access)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:10: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y())
        : "l"(src), "r"((int)do_access));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<8, LoadPolicy::CacheStreaming>(void* dst, const void* src)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:11: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y())
                 : "l"(src));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<8, LoadPolicy::CacheStreaming>(void* dst,
                                                                const void* src,
                                                                bool do_access)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:12: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y())
        : "l"(src), "r"((int)do_access));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<4>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:13: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.ca.u32 {%0}, [%1];\n" : "=r"(*data) : "l"(src));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<4>(void* dst, const void* src, bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:14: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.global.u32 {%0}, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "l"(src), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<4, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:15: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cg.u32 {%0}, [%1];\n" : "=r"(*data) : "l"(src));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<4, LoadPolicy::CacheGlobal>(void* dst,
                                                             const void* src,
                                                             bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:16: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.global.cg.u32 {%0}, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "l"(src), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<4, LoadPolicy::CacheStreaming>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:17: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cs.u32 {%0}, [%1];\n" : "=r"(*data) : "l"(src));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<4, LoadPolicy::CacheStreaming>(void* dst,
                                                                const void* src,
                                                                bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:18: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.global.cs.u32 {%0}, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "l"(src), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<2>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:19: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.ca.u16 {%0}, [%1];\n" : "=h"(*data) : "l"(src));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<2>(void* dst, const void* src, bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:20: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.u16 %0, 0;\n"
        "\t@p ld.global.u16 {%0}, [%1];\n"
        "}\n"
        : "=h"(*data)
        : "l"(src), "r"((int)do_access));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<2, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:21: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cg.u16 {%0}, [%1];\n" : "=h"(*data) : "l"(src));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<2, LoadPolicy::CacheGlobal>(void* dst,
                                                             const void* src,
                                                             bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:22: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.u16 %0, 0;\n"
        "\t@p ld.global.cg.u16 {%0}, [%1];\n"
        "}\n"
        : "=h"(*data)
        : "l"(src), "r"((int)do_access));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_global<2, LoadPolicy::CacheStreaming>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:23: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cs.u16 {%0}, [%1];\n" : "=h"(*data) : "l"(src));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_global<2, LoadPolicy::CacheStreaming>(void* dst,
                                                                const void* src,
                                                                bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:24: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.u16 %0, 0;\n"
        "\t@p ld.global.cs.u16 {%0}, [%1];\n"
        "}\n"
        : "=h"(*data)
        : "l"(src), "r"((int)do_access));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

/////////// Load Shared ///////////
namespace internal {

#ifdef PTX_AVAILABLE
__dpct_inline__ unsigned convert_to_shared(const void* ptr)
{
#if __SYCLCC_VER_MAJOR__ >= 11
    // In SYCL 11 we have a builtin intrinsic
    return __cvta_generic_to_shared(ptr);
#else
    unsigned ret_val;
    asm volatile(
        "{\n"
        "\t.reg .u64 p1;\n"
        "\tcvta.to.shared.u64 p1, %1\n"
        "\tcvt.u32.u64 %0, p1;\n"
        "}\n"
        : "=r"(ret_val)
        : "l"(ptr));
    return ret_val;
#endif
}
#endif

}  // namespace internal

template <>
__dpct_inline__ void load_shared<16>(void* dst, const void* src)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    /*
    DPCT1053:25: Migration of device assembly code is not supported.
    */
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
                 : "r"(src_shr));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_shared<16>(void* dst, const void* src, bool do_access)
{
    sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    /*
    DPCT1053:26: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y()), "=r"(data[0].z()), "=r"(data[0].w())
        : "r"(src_shr), "r"((int)do_access));
#else
    const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_shared<8>(void* dst, const void* src)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    /*
    DPCT1053:27: Migration of device assembly code is not supported.
    */
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x()), "=r"(data[0].y())
                 : "r"(src_shr));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_shared<8>(void* dst, const void* src, bool do_access)
{
    sycl::uint2* data = reinterpret_cast<sycl::uint2*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    /*
    DPCT1053:28: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.shared.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x()), "=r"(data[0].y())
        : "r"(src_shr), "r"((int)do_access));
#else
    const sycl::uint2* src_cast = reinterpret_cast<const sycl::uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
#endif
}

template <>
__dpct_inline__ void load_shared<4>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    /*
    DPCT1053:29: Migration of device assembly code is not supported.
    */
    asm volatile("ld.shared.u32 {%0}, [%1];\n" : "=r"(*data) : "r"(src_shr));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__dpct_inline__ void load_shared<4>(void* dst, const void* src, bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    /*
    DPCT1053:30: Migration of device assembly code is not supported.
    */
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.shared.u32 %0, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "r"(src_shr), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

/////////// Store Global ///////////

template <>
__dpct_inline__ void store_global<16>(void* dst, const void* src)
{
    const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:31: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(data[0].x()), "r"(data[0].y()), "r"(data[0].z()), "r"(data[0].w())
                 : "memory");
#else
    sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<16, StorePolicy::CacheGlobal>(void* dst, const void* src)
{
    const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:32: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(data[0].x()), "r"(data[0].y()), "r"(data[0].z()), "r"(data[0].w())
                 : "memory");
#else
    sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<16, StorePolicy::CacheStreaming>(void* dst, const void* src)
{
    const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:33: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(data[0].x()), "r"(data[0].y()), "r"(data[0].z()), "r"(data[0].w())
                 : "memory");
#else
    sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<8>(void* dst, const void* src)
{
    const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:34: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "l"(dst), "r"(data[0].x()), "r"(data[0].y()));
#else
    sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<8, StorePolicy::CacheGlobal>(void* dst, const void* src)
{
    const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:35: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cg.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "l"(dst), "r"(data[0].x()), "r"(data[0].y()));
#else
    sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<8, StorePolicy::CacheStreaming>(void* dst, const void* src)
{
    const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:36: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "l"(dst), "r"(data[0].x()), "r"(data[0].y()));
#else
    sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<4>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:37: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.wb.u32 [%0], %1;\n" : : "l"(dst), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<4, StorePolicy::CacheGlobal>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:38: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cg.u32 [%0], %1;\n" : : "l"(dst), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_global<4, StorePolicy::CacheStreaming>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    /*
    DPCT1053:39: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cs.u32 [%0], %1;\n" : : "l"(dst), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

/////////// Store Shared ///////////

template <>
__dpct_inline__ void store_shared<16>(void* dst, const void* src)
{
    const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
#ifdef PTX_AVAILABLE
    unsigned dst_int = internal::convert_to_shared(dst);

    /*
    DPCT1053:40: Migration of device assembly code is not supported.
    */
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(dst_int), "r"(data[0].x()), "r"(data[0].y()), "r"(data[0].z()), "r"(data[0].w()));
#else
    sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_shared<8>(void* dst, const void* src)
{
    const sycl::uint2* data = reinterpret_cast<const sycl::uint2*>(src);
#ifdef PTX_AVAILABLE
    unsigned dst_int = internal::convert_to_shared(dst);

    /*
    DPCT1053:41: Migration of device assembly code is not supported.
    */
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "r"(dst_int), "r"(data[0].x()), "r"(data[0].y()));
#else
    sycl::uint2* dst_cast = reinterpret_cast<sycl::uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__dpct_inline__ void store_shared<4>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    unsigned dst_int = internal::convert_to_shared(dst);

    /*
    DPCT1053:42: Migration of device assembly code is not supported.
    */
    asm volatile("st.shared.u32 [%0], %1;\n" : : "r"(dst_int), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

/////////// Asynchronous Memory Copy ///////////

#ifdef ASYNC_COPY_AVAILABLE
template <int AccessSize>
__device__ __forceinline__ void memcpy_async(void* shr, const void* gbl)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                 :
                 : "r"(shr_int), "l"(gbl), "n"(AccessSize));
}

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_nop(void* shr, const void* gbl, bool predicate)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n"
        :
        : "r"((int)predicate), "r"(shr_int), "l"(gbl), "n"(AccessSize));
}

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_zero(void* shr, const void* gbl, bool predicate)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (predicate ? AccessSize : 0);

    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                 :
                 : "r"(shr_int), "l"(gbl), "n"(AccessSize), "r"(bytes_to_copy));
}

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_zero_nop(void* shr,
                                                      const void* gbl,
                                                      bool zero_predicate,
                                                      bool nop_predicate)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (zero_predicate ? AccessSize : 0);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
        "}\n"
        :
        : "r"((int)nop_predicate), "r"(shr_int), "l"(gbl), "n"(AccessSize), "r"(bytes_to_copy));
}

// Cache global variants. Separate interface to require deliberate use of them.
__device__ __forceinline__ void memcpy_async_cg(void* shr, const void* gbl)
{
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(shr_int), "l"(gbl));
}

__device__ __forceinline__ void memcpy_async_nop_cg(void* shr, const void* gbl, bool predicate)
{
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], 16;\n"
        "}\n"
        :
        : "r"((int)predicate), "r"(shr_int), "l"(gbl));
}

__device__ __forceinline__ void memcpy_async_zero_cg(void* shr, const void* gbl, bool predicate)
{
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (predicate ? 16 : 0);

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(shr_int), "l"(gbl), "r"(bytes_to_copy));
}

__device__ __forceinline__ void memcpy_async_zero_nop_cg(void* shr,
                                                         const void* gbl,
                                                         bool zero_predicate,
                                                         bool nop_predicate)
{
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (zero_predicate ? 16 : 0);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], 16, %3;\n"
        "}\n"
        :
        : "r"((int)nop_predicate), "r"(shr_int), "l"(gbl), "r"(bytes_to_copy));
}

__device__ __forceinline__ void memcpy_async_fence() { asm volatile("cp.async.commit_group;\n"); }

template <int stages>
__device__ __forceinline__ void memcpy_async_wait()
{
    static_assert(stages <= 8);

    asm volatile("cp.async.wait_group %0;\n" : : "n"(stages));
}

// TODO: The tail complete should be a known compile time artifact, should try and induce this
// without all of the branches from the call-site. This is a hacky solution.
template <>
__device__ __forceinline__ void tail_complete_wait<1>(int remaining_stages)
{
    if (remaining_stages == 0) memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<2>(int remaining_stages)
{
    if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<3>(int remaining_stages)
{
    if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<4>(int remaining_stages)
{
    if (remaining_stages == 3)
        memcpy_async_wait<3>();
    else if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<5>(int remaining_stages)
{
    if (remaining_stages == 4)
        memcpy_async_wait<4>();
    else if (remaining_stages == 3)
        memcpy_async_wait<3>();
    else if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<6>(int remaining_stages)
{
    if (remaining_stages == 5)
        memcpy_async_wait<5>();
    else if (remaining_stages == 4)
        memcpy_async_wait<4>();
    else if (remaining_stages == 3)
        memcpy_async_wait<3>();
    else if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}
#endif

}  // namespace mem_access
