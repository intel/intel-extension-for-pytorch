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

#include <stdint.h>

#ifdef BF16_AVAILABLE
#endif

namespace conversion {

// Basic primitive for constructing conversions
template <typename TO, typename FROM>
DS_D_INLINE TO to(FROM val)
{
    return to(val);
}

// Specializations

/********************* Identity Conversions *********************/
/*
Identity conversions are useful in templated functions where we might have
a fixed destination type. For example, I might have a kernel that accepts
sycl::half, __nv_bfloat16, and float but always want to do the core computation
at floating point:

T mem_value = input[idx];
float compute_value = conversion::to<float, T>(mem_value);

In practice, we should be able to elide the second template parameter:
float compute_val = conversion::to<float>(mem_value);

In this case, we need an implementation to handle the T = float case

NOTE: The type inferencing system appears to be unable to handle inferring the first
template parameter, even in the trivial case.
*/

// Floating point types
template <>
DS_D_INLINE double to(double val)
{
    return val;
}
template <>
DS_D_INLINE float to(float val)
{
    return val;
}
template <>
DS_D_INLINE sycl::half to(sycl::half val)
{
    return val;
}
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(sycl::ext::oneapi::bfloat16 val)
{
    return val;
}
#endif

// Integer types
template <>
DS_D_INLINE int8_t to(int8_t val)
{
    return val;
}
template <>
DS_D_INLINE uint8_t to(uint8_t val)
{
    return val;
}
template <>
DS_D_INLINE int16_t to(int16_t val)
{
    return val;
}
template <>
DS_D_INLINE uint16_t to(uint16_t val)
{
    return val;
}
template <>
DS_D_INLINE int32_t to(int32_t val)
{
    return val;
}
template <>
DS_D_INLINE uint32_t to(uint32_t val)
{
    return val;
}
template <>
DS_D_INLINE int64_t to(int64_t val)
{
    return val;
}
template <>
DS_D_INLINE uint64_t to(uint64_t val)
{
    return val;
}

// TODO: evaluate if we want bools

/*********************  To Double Conversions *********************/

// * to double variants

// Would normally like to not use C cast, but this is an important enough conversion
// to keep
template <>
DS_D_INLINE double to(float val)
{
#ifdef PTX_AVAILABLE
    double ret_val;
    /*
    DPCT1053:0: Migration of device assembly code is not supported.
    */
    asm("ctv.rn.f64.f32 %0, %1;\n" : "=d"(ret_val) : "f"(val));
    return ret_val;
#else
    return double(val);
#endif
}
// Note: there is a CVT instruction for sycl::half -> double, but there's no inline interface
// for passing a single half value
template <>
DS_D_INLINE double to(sycl::half val)
{
    return to<double>(
        sycl::vec<sycl::half, 1>{val}.convert<float, sycl::rounding_mode::automatic>()[0]);
}
template <>
DS_D_INLINE double to(int64_t val)
{
    return sycl::vec<long long, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(int32_t val)
{
    return sycl::vec<int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(int16_t val)
{
    return sycl::vec<int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(int8_t val)
{
    return sycl::vec<int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint64_t val)
{
    return sycl::vec<unsigned long long, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint32_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint16_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint8_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}

// Same applies here
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE double to(sycl::ext::oneapi::bfloat16 val)
{
    return to<double>(static_cast<float>(val));
}
#endif

/*********************  To Float Conversions *********************/

template <>
DS_D_INLINE float to(double val)
{
    return sycl::vec<double, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<float, sycl::rounding_mode::automatic>()[0];
}
template <>
DS_D_INLINE float to(int64_t val)
{
    return sycl::vec<long long, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(int32_t val)
{
    return sycl::vec<int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(int16_t val)
{
    return sycl::vec<int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(int8_t val)
{
    return sycl::vec<int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint64_t val)
{
    return sycl::vec<unsigned long long, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint32_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint16_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint8_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE float to(sycl::ext::oneapi::bfloat16 val)
{
    return static_cast<float>(val);
}
#endif

/*********************  To Float2 Conversions *********************/
template <>
DS_D_INLINE sycl::float2 to(sycl::half2 val)
{
    return val.convert<float, sycl::rounding_mode::automatic>();
}

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE sycl::float2 to(sycl::marray<sycl::ext::oneapi::bfloat16, 2> val)
{
    return sycl::float2(val[0], val[1]);
}
#endif

/*********************  To Half Conversions *********************/
template <>
DS_D_INLINE sycl::half to(double val)
{
    return sycl::half(val);
}
template <>
DS_D_INLINE sycl::half to(float val)
{
    return sycl::vec<float, 1>{val}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}
template <>
DS_D_INLINE sycl::half to(int64_t val)
{
    return sycl::vec<long long, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(int32_t val)
{
    return sycl::vec<int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(int16_t val)
{
    return sycl::vec<short, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(int8_t val)
{
    return sycl::vec<int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint64_t val)
{
    return sycl::vec<unsigned long long, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint32_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint16_t val)
{
    return sycl::vec<unsigned short, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint8_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
DS_D_INLINE sycl::half to(sycl::ext::oneapi::bfloat16 val)
{
    return to<sycl::half>(to<float>(val));
}
#endif

/*********************  To Half2 Conversions *********************/
template <>
DS_D_INLINE sycl::half2 to(sycl::float2 val)
{
    return val.convert<sycl::half, sycl::rounding_mode::rte>();
}
template <>
DS_D_INLINE sycl::half2 to(float val)
{
    return sycl::float2{val, val}.convert<sycl::half, sycl::rounding_mode::rte>();
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
DS_D_INLINE sycl::half2 to(sycl::marray<sycl::ext::oneapi::bfloat16, 2> val)
{
    return to<sycl::half2>(to<sycl::float2>(val));
}
#endif

/*********************  To BF16 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(double val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(float val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(int64_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(int32_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(int16_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(int8_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(uint64_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(uint32_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(uint16_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
template <>
DS_D_INLINE sycl::ext::oneapi::bfloat16 to(uint8_t val)
{
    return sycl::ext::oneapi::bfloat16(val);
}
#endif

/*********************  To BF162 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE sycl::marray<sycl::ext::oneapi::bfloat16, 2> to(sycl::float2 val)
{
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(val[0], val[1]);
}
template <>
DS_D_INLINE sycl::marray<sycl::ext::oneapi::bfloat16, 2> to(float val)
{
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(val, val);
}
template <>
DS_D_INLINE sycl::marray<sycl::ext::oneapi::bfloat16, 2> to(sycl::half2 val)
{
    return to<sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(to<sycl::float2>(val));
}
#endif

/*********************  To INT64_T Conversions *********************/
template <>
DS_D_INLINE int64_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int64_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int64_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<long long, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int64_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<long long, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To INT32_T Conversions *********************/
template <>
DS_D_INLINE int32_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int32_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int32_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int32_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<int, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To INT16_T Conversions *********************/
template <>
DS_D_INLINE int16_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int16_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int16_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int16_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<int, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To INT8_T Conversions *********************/
template <>
DS_D_INLINE int8_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int8_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int8_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int8_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<int, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To UINT64_T Conversions *********************/
template <>
DS_D_INLINE uint64_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint64_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint64_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint64_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To UINT32_T Conversions *********************/
template <>
DS_D_INLINE uint32_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint32_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint32_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint32_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<unsigned, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To UINT16_T Conversions *********************/
template <>
DS_D_INLINE uint16_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint16_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint16_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint16_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<unsigned, sycl::rounding_mode::rte>()[0];
}
#endif

/*********************  To UINT8_T Conversions *********************/
template <>
DS_D_INLINE uint8_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint8_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint8_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint8_t to(sycl::ext::oneapi::bfloat16 val)
{
    return sycl::vec<float, 1>(val).convert<unsigned, sycl::rounding_mode::rte>()[0];
}
#endif

}  // namespace conversion
