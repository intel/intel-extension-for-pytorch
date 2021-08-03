#pragma once

#include <CL/sycl.hpp>
#include <c10/core/Device.h>

#include <utils/Macros.h>

IPEX_API bool is_profiler_enabled();

IPEX_API void dpcpp_mark(std::string name, cl::sycl::event& event);

IPEX_API void dpcpp_mark(
    std::string name,
    cl::sycl::event& start_event,
    cl::sycl::event& end_event);

IPEX_API void dpcpp_log(std::string name, cl::sycl::event& event);

IPEX_API void dpcpp_log(
    std::string name,
    cl::sycl::event& start_event,
    cl::sycl::event& end_event);

IPEX_API void reportMemoryUsage(
    void* ptr,
    int64_t alloc_size,
    at::DeviceIndex device_id);
