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

/*
Centralized header file for preprocessor macros and constants
used throughout the codebase.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.h>

#ifdef BF16_AVAILABLE
#endif

#define DS_HD_INLINE __dpct_inline__
#define DS_D_INLINE __dpct_inline__


// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#define HALF_PRECISION_AVAILABLE = 1
// #define PTX_AVAILABLE


inline int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}
