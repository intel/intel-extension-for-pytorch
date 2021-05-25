#pragma once


namespace xpu {
namespace dpcpp {

void dpcppMemoryScale(
    void* dst,
    const void* src,
    size_t n_elements,
    float alpha);

void dpcppMemoryScale1(
    void* dst,
    const void* src,
    size_t n_elements,
    const double eps);

void dpcppMemoryScale2(
    void* dst,
    const void* src,
    size_t n_elements,
    const float alpha,
    const double eps);

} // namespace dpcpp
} // namespace xpu
