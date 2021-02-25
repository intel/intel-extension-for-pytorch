#include <core/Memory.h>
#include <utils/Profiler.h>
#include <core/CachingHostAllocator.h>
#include <core/Stream.h>

namespace at {
namespace dpcpp {

static void memcpyHostToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    DPCPP::queue& dpcpp_queue) {
  if (n_bytes == 0)
    return;

  auto e = dpcpp_queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else {
    dpcpp_recordEventInCachingHostAllocator(const_cast<void*>(src), e);
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_Q_FORCE_SYNC(dpcpp_queue);
}

static void memcpyDeviceToHost(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    DPCPP::queue& dpcpp_queue) {
  if (n_bytes == 0)
    return;

  auto e = dpcpp_queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else {
    dpcpp_recordEventInCachingHostAllocator(const_cast<void*>(dst), e);
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_Q_FORCE_SYNC(dpcpp_queue);
}

static void memcpyDeviceToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    DPCPP::queue& dpcpp_queue) {
  if (n_bytes == 0)
    return;

  auto e = dpcpp_queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_Q_FORCE_SYNC(dpcpp_queue);
}

static void memsetDevice(
    void* dst,
    int value,
    size_t n_bytes,
    bool async,
    DPCPP::queue& dpcpp_queue) {
  auto e = dpcpp_queue.memset(dst, value, n_bytes);
  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_Q_FORCE_SYNC(dpcpp_queue);
}

void dpcppMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, false, dpcpp_queue);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, false, dpcpp_queue);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, false, dpcpp_queue);
      break;
    default:
      throw std::runtime_error("Unknown dpcpp memory kind");
  }
}

void dpcppMemcpyAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, true, dpcpp_queue);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, true, dpcpp_queue);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, true, dpcpp_queue);
      break;
    default:
      throw std::runtime_error("Unknown dpcpp memory kind");
  }
}

void dpcppMemset(void* data, int value, size_t n_bytes) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  memsetDevice(data, value, n_bytes, false, dpcpp_queue);
}

void dpcppMemsetAsync(void* data, int value, size_t n_bytes) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  memsetDevice(data, value, n_bytes, true, dpcpp_queue);
}

} // namespace dpcpp
} // namespace at
