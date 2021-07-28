#include <runtime/Memory.h>
#include <runtime/Queue.h>
#include <runtime/CachingHostAllocator.h>
#include <utils/Profiler.h>

namespace xpu {
namespace dpcpp {

void memcpyHostToDevice(void* dst, const void* src, size_t n_bytes, bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else {
    CachingHostAllocator::Instance()->recordEvent(const_cast<void*>(src), e);
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_FORCE_SYNC(e);
}

void memcpyDeviceToHost(void* dst, const void* src, size_t n_bytes, bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else {
    CachingHostAllocator::Instance()->recordEvent(const_cast<void*>(dst), e);
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_FORCE_SYNC(e);
}

void memcpyDeviceToDevice(void* dst, const void* src, size_t n_bytes, bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_FORCE_SYNC(e);
}

void memsetDevice(void* dst, int value, size_t n_bytes, bool async) {
  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memset(dst, value, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_FORCE_SYNC(e);
}

} // namespace dpcpp
} // namespace xpu
