#pragma once

#include <CL/sycl.hpp>
#include <c10/core/Device.h>
#include "Macros.h"

namespace xpu {

IPEX_API bool is_profiler_enabled();

IPEX_API void profiler_record(std::string name, cl::sycl::event& event);

IPEX_API void profiler_record(
    std::string name,
    cl::sycl::event& start_event,
    cl::sycl::event& end_event);

} // namespace xpu
