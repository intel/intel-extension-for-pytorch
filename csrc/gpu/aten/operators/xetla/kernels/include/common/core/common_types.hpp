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
#include <cstdint>

namespace gpu::xetla {
enum class gpu_arch : uint8_t {
  XeLpg = 0,
  XeHpg = 1,
  XeHpc = 2,
  XeHpc_vg = 3,
  Xe2Lpg = 4,
  Xe2Hpg = 5,
  XeLast
};

template <gpu_arch arch_tag>
inline constexpr bool valid_xe_arch_tag = (arch_tag < gpu_arch::XeLast);

enum class mma_engine : uint8_t { xmx = 0, fpu = 1 };

enum class grf_mode : uint8_t { normal_grf = 0, double_grf = 1 };

enum class mem_layout : uint8_t { row_major = 0, col_major = 1 };

enum class quant_mode : uint8_t {
  // Asymmetric quantization with zero point of the same dtype as the weight
  I4_ASYM = 0,

  // Symmetric quantization without zero point
  I4_SYM = 1,

  // Asymmetric quantization with zero point of the same dtype as the input
  I4_ASYM_FP_ZERO = 2
};

// for marlin dequantization
enum class DequantMode : uint8_t {
  Basic = 0,
  FastInterleaved = 1,
  FastInterleavedWithScaleMerge = 2
};

struct quant_info {
  quant_mode quant_mode;
  uint32_t dequant_s;
  mem_layout weight_mem_layout;
};

enum class fp8_format : uint8_t {
  E4M3 = 0,
  E5M2 = 1,
};

} // namespace gpu::xetla
