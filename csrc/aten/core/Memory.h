#pragma once

#include <utils/DPCPP.h>
#include <core/Stream.h>

DPCPP_DEF_K1(memory_copy);

namespace xpu {
namespace dpcpp {

enum dpcppMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

void dpcppMemcpy(void* dst, const void* src, size_t n_bytes, dpcppMemcpyKind kind);

void dpcppMemcpyAsync(void* dst, const void* src, size_t n_bytes, dpcppMemcpyKind kind);

void dpcppMemset(void* data, int value, size_t n_bytes);

void dpcppMemsetAsync(void* data, int value, size_t n_bytes);

} // namespace dpcpp
} // namespace at
