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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <c10/core/Device.h>
#include "Macros.h"

namespace xpu {

/// Return whether enable profiler.
IPEX_API bool is_profiler_enabled();

/// Use profiler to record event.
/// @param name: name for the profiler recording.
/// @param event: sycl event for recording.
IPEX_API void profiler_record(std::string name, sycl::event& event);

/// Use profiler to record event.
/// @param name: name for the profiler recording.
/// @param start_event: sycl start event for recording.
/// @param end_event: sycl end event for recording.
IPEX_API void profiler_record(
    std::string name,
    sycl::event& start_event,
    sycl::event& end_event);

} // namespace xpu
