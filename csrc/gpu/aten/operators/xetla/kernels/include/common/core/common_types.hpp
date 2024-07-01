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
enum class gpu_arch : uint8_t { XeLpg = 0, XeHpg = 1, XeHpc = 2 };
inline constexpr bool arch_has_xmx(gpu_arch arch) {
  return arch >= gpu_arch::XeHpg;
}
inline constexpr bool arch_has_2d_load_store(gpu_arch arch) {
  return arch >= gpu_arch::XeHpc;
}

enum class grf_mode : uint8_t { normal = 0, double_grf = 1 };

enum class mem_layout : uint8_t { row_major = 0, col_major = 1 };
} // namespace gpu::xetla
