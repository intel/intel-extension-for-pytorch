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

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include "Macros.h"

namespace xpu {
namespace dpcpp {

/// Get a tensor from a united shared memory.
/// @param src: a pointer of united shared memory.
/// @param stype: date type.
/// @param shape: shape.
/// @param strides: strides.
/// @param device_id: device id.
/// @returns: Tensor.
IPEX_API at::Tensor fromUSM(
    void* src,
    const at::ScalarType stype,
    at::IntArrayRef shape,
    c10::optional<at::IntArrayRef> strides = c10::nullopt,
    const at::DeviceIndex device_id = -1);

/// Get a pointer of united shared memory from a tensor.
/// @param src: Tensor.
/// @returns: a pointer of united shared memory.
IPEX_API void* toUSM(const at::Tensor& src);

} // namespace dpcpp
} // namespace xpu
