#pragma once

#include <utils/DPCPP.h>

namespace xpu {
namespace dpcpp {

void memcpyHostToDevice(void* dst, const void* src, size_t n_bytes, bool async);

void memcpyDeviceToHost(void* dst, const void* src, size_t n_bytes, bool async);

void memcpyDeviceToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async);

void memsetDevice(void* dst, int value, size_t n_bytes, bool async);

} // namespace dpcpp
} // namespace xpu
