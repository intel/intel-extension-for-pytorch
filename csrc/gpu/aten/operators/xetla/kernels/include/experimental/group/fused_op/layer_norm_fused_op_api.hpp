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

#include <subgroup/subgroup.hpp>

namespace gpu::xetla {

/// @brief
///
enum class ln_fwd_fused_kind : uint8_t {
  none = 0,
  bias_dropout_resAdd_ln = 1,
  ln_dropout = 2,
  // fused with random number generator kernel
  bias_rng_dropout_resAdd_ln = 3,
  // fused with random number generator kernel
  ln_rng_dropout = 4,
};

/// @brief
///
enum class ln_bwd_fused_kind : uint8_t {
  none = 0,
  bias_dropout_resAdd_ln = 1,
  ln_dropout_gradAdd = 2,
  ln_dropout = 3,
};

} // namespace gpu::xetla
