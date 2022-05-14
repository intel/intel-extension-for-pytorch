#pragma once

#include <CL/sycl.hpp>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

namespace xpu {
namespace dpcpp {

void parallel_for_setup(
    int64_t n,
    int64_t& tileSize,
    int64_t& rng,
    int64_t& GRange);

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& GRange0,
    int64_t& GRange1);

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& tileSize2,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& rng2,
    int64_t& GRange0,
    int64_t& GRange1,
    int64_t& GRange2);

cl::sycl::event queue_barrier(cl::sycl::queue& queue);
cl::sycl::event queue_barrier(
    cl::sycl::queue& queue,
    std::vector<cl::sycl::event>& events);

} // namespace dpcpp
} // namespace xpu
