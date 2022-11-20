/*******************************************************************************
 * Copyright 2016-2022 Intel Corporation
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

/** @file */

#pragma once

#include "Macros.h"

namespace xpu {

/// \enum FP32_MATH_MODE
/// \brief specifies the available DPCCP packet types
enum IPEX_API FP32_MATH_MODE {
  FP32 = 0, ///< set floating-point math mode to FP32.
  TF32 = 1, ///< set floating-point math mode to TF32.
  BF32 = 2, ///< set floating-point math mode to BF32.
  FP32_MATH_MODE_MAX = BF32 ///< set floating-point math mode.
};
static const char* IPEX_API FP32_MATH_MODE_STR[]{"FP32", "TF32", "BF32"};

/// Get Math Mode Setting Status.
IPEX_API FP32_MATH_MODE get_fp32_math_mode();

/// Enable or disable implicit data type conversion.
/// If mode is FP32MathMode.FP32 which means to disable the oneDNN fpmath mode.
/// If mode is FP32MathMode.TF32 which means to enable the oneDNN fpmath mode by
/// down converting to tf32 implicitly If mode is FP32MathMode.BF32 which means
/// to enable the oneDNN fpmath mode by down converting to bfloat16 implicitly.
/// @param mode (FP32MathMode): Only works for ``FP32MathMode.FP32``,
/// ``FP32MathMode.TF32`` and ``FP32MathMode.BF32``.
///     oneDNN fpmath mode will be disabled by default if dtype is set to
///     ``FP32MathMode.FP32``. The implicit FP32 to TF32 data type conversion
///     will be enabled if dtype is set to ``FP32MathMode.TF32`. The implicit
///     FP32 to BF16 data type conversion will be enabled if dtype is set to
///     ``FP32MathMode.BF32`.
IPEX_API bool set_fp32_math_mode(FP32_MATH_MODE mode);

} // namespace xpu
