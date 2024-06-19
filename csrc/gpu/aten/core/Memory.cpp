#include <ATen/detail/XPUHooksInterface.h>
#include <aten/operators/comm/ScalarType.h>
#include <core/Memory.h>
#include <runtime/Memory.h>

namespace torch_ipex::xpu {
namespace dpcpp {

void dpcppMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind) {
  switch (kind) {
    case HostToDevice:
      // for synchronous copy, the context of host data pointer is unnecessary.
      memcpyHostToDevice(dst, src, n_bytes, false, nullptr);
      break;
    case DeviceToHost:
      // for synchronous copy, the context of host data pointer is unnecessary.
      memcpyDeviceToHost(dst, src, n_bytes, false, nullptr);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, false);
      break;
    default:
      TORCH_CHECK(false, "Unknown dpcpp memory kind");
  }
}

void dpcppMemcpyAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind,
    const void* hctx) {
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(
          dst,
          src,
          n_bytes,
          true,
          hctx,
          at::detail::getXPUHooks().isPinnedPtr(src));
      break;
    case DeviceToHost:
      memcpyDeviceToHost(
          dst,
          src,
          n_bytes,
          true,
          hctx,
          at::detail::getXPUHooks().isPinnedPtr(dst));
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, true);
      break;
    default:
      TORCH_CHECK(false, "Unknown dpcpp memory kind");
  }
}

void dpcppMemset(void* data, int value, size_t n_bytes) {
  memsetDevice(data, value, n_bytes, false);
}

void dpcppMemsetAsync(void* data, int value, size_t n_bytes) {
  memsetDevice(data, value, n_bytes, true);
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
