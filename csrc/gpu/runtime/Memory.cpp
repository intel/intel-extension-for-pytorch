#include <runtime/CachingHostAllocator.h>
#include <runtime/Memory.h>
#include <runtime/Utils.h>
#include <utils/Profiler.h>

namespace xpu {
namespace dpcpp {

void memcpyHostToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = dpcppGetCurrentQueue();
  sycl::event e;

  if (!async) {
    e = queue.memcpy(dst, src, n_bytes);
    e.wait();
  } else {
    // For async H2D copy, it wll check the src is allocated by SYCL API or
    // system. For system allocated memory, the H2D will firstly copy data
    // from the pageable memory to the unpageable memory, then execute H2D.
    if (CachingHostAllocator::Instance()->isHostPtr(src)) {
      e = queue.memcpy(dst, src, n_bytes);
      CachingHostAllocator::Instance()->recordEvent(const_cast<void*>(src), e);
    } else {
      void* src_host_alloc = nullptr;
      auto err =
          CachingHostAllocator::Instance()->malloc(&src_host_alloc, n_bytes);
      if (err != DPCPP_SUCCESS) {
        throw std::runtime_error(
            "Fail to allocate host memory from IPEX CachingHostAllocator");
      }

      // copy from the pageable memory to the unpageable memory
      std::memcpy(src_host_alloc, src, n_bytes);
      e = queue.memcpy(dst, src_host_alloc, n_bytes);
      CachingHostAllocator::Instance()->recordEvent(
          const_cast<void*>(src_host_alloc), e);

      // obviously release the allocated unpageable memory to let its associated
      // block can be recycled upon its recorded events are all completed
      // because the src_host_alloc is not mantained by any unique ptr, so no
      // one can free it unless here
      CachingHostAllocator::Instance()->release(src_host_alloc);
    }
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

void memcpyDeviceToHost(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = dpcppGetCurrentQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else {
    CachingHostAllocator::Instance()->recordEvent(const_cast<void*>(dst), e);
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

void memcpyDeviceToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = dpcppGetCurrentQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

void memsetDevice(void* dst, int value, size_t n_bytes, bool async) {
  auto& queue = dpcppGetCurrentQueue();
  auto e = queue.memset(dst, value, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

} // namespace dpcpp
} // namespace xpu
