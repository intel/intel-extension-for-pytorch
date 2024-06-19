#pragma once

#include <utils/DPCPP.h>

namespace torch_ipex::xpu {
namespace dpcpp {

enum dpcppMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

void dpcppMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind);

// hctx is the context of host data ptr, it is mandatory in the following two
// scenarios:
//   1. copy device to host and host data ptr is a pinned memory;
//   2. copy host to device and host data ptr is a pinned memory;
void dpcppMemcpyAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind,
    const void* hctx = nullptr);

void dpcppMemset(void* data, int value, size_t n_bytes);

void dpcppMemsetAsync(void* data, int value, size_t n_bytes);

} // namespace dpcpp
} // namespace torch_ipex::xpu
