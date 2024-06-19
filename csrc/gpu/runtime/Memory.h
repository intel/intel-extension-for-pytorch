#pragma once

#include <utils/DPCPP.h>

namespace torch_ipex::xpu {
namespace dpcpp {

void memcpyHostToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    const void* hctx,
    bool is_pinned = false);

void memcpyDeviceToHost(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    const void* hctx,
    bool is_pinned = false);

void memcpyDeviceToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async);

void memsetDevice(void* dst, int value, size_t n_bytes, bool async);

} // namespace dpcpp
} // namespace torch_ipex::xpu
