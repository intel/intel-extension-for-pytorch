#pragma once

#include <CL/sycl.hpp>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

namespace xpu {
namespace dpcpp {

cl::sycl::event queue_barrier(cl::sycl::queue& queue);
cl::sycl::event queue_barrier(
    cl::sycl::queue& queue,
    std::vector<cl::sycl::event>& events);

} // namespace dpcpp
} // namespace xpu
