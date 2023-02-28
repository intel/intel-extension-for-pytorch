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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <c10/core/Stream.h>

namespace xpu {

/// Get a sycl queue from a c10 stream. Generate a dpcpp stream from c10 stream,
/// and get dpcpp queue.
/// @param stream: c10 stream.
/// @returns: dpcpp queue.
IPEX_API sycl::queue& get_queue_from_stream(c10::Stream stream);

} // namespace xpu
