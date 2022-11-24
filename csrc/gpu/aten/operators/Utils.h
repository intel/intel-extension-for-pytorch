#pragma once

#include <utils/DPCPP.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dtype_convert_by_scalar(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements);

DPCPP_HOST sycl::event dpcpp_q_barrier(sycl::queue& q);
DPCPP_HOST sycl::event dpcpp_q_barrier(
    sycl::queue& q,
    std::vector<sycl::event>& events);

} // namespace AtenIpexTypeXPU
} // namespace at
