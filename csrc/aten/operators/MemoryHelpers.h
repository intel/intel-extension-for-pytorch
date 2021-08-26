#pragma once

#include <utils/DPCPP.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dpcppMemoryScale(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    float alpha);

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dpcppMemoryScale1(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    const double eps);

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dpcppMemoryScale2(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    const float alpha,
    const double eps);

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dtype_convert_by_scalar(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements);

} // namespace AtenIpexTypeXPU
} // namespace at
