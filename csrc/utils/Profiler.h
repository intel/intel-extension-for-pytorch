#pragma once

#include <CL/sycl.hpp>
#include <c10/core/Device.h>

#include <utils/Macros.h>

bool is_profiler_enabled();

void dpcpp_mark(std::string name, cl::sycl::event& event);

void dpcpp_mark(
    std::string name,
    cl::sycl::event& start_event,
    cl::sycl::event& end_event);

void dpcpp_log(std::string name, cl::sycl::event& event);

void dpcpp_log(
    std::string name,
    cl::sycl::event& start_event,
    cl::sycl::event& end_event);

void reportMemoryUsage(
    void* ptr,
    int64_t alloc_size,
    at::DeviceIndex device_id);
