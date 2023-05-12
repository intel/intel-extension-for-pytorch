#pragma once

#include <utils/DPCPP.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

sycl::event dpcpp_q_barrier(sycl::queue& q);
sycl::event dpcpp_q_barrier(sycl::queue& q, std::vector<sycl::event>& events);

} // namespace AtenIpexTypeXPU
} // namespace at
