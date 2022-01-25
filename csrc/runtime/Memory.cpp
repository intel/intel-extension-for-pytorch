#include <aten/operators/comm/ScalarType.h>
#include <runtime/CachingHostAllocator.h>
#include <runtime/Memory.h>
#include <runtime/Queue.h>
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

  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else {
    CachingHostAllocator::Instance()->recordEvent(const_cast<void*>(src), e);
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

  auto& queue = getCurrentQueue()->getDpcppQueue();
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

  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

void memsetDevice(void* dst, int value, size_t n_bytes, bool async) {
  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.memset(dst, value, n_bytes);

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

template <class T>
void fillDevice(T* dst, T value, size_t n_elems, bool async) {
  auto& queue = getCurrentQueue()->getDpcppQueue();
  auto e = queue.submit(DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      *(dst + item_id) = value;
    };
    __cgh.parallel_for(DPCPP::range<1>(n_elems), kfn);
  });

  if (!async) {
    e.wait();
  }

  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
}

#define DEFINE_CAST(T, name)                                     \
  template <>                                                    \
  void fillDevice(T* dst, T value, size_t n_elems, bool async) { \
    auto& queue = getCurrentQueue()->getDpcppQueue();            \
    auto e = queue.submit(DPCPP_Q_CGF(__cgh) {                   \
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {           \
        *(dst + item_id) = value;                                \
      };                                                         \
      __cgh.parallel_for(DPCPP::range<1>(n_elems), kfn);         \
    });                                                          \
                                                                 \
    if (!async) {                                                \
      e.wait();                                                  \
    }                                                            \
                                                                 \
    dpcpp_log("dpcpp_kernel", e);                                \
    DPCPP_E_SYNC_FOR_DEBUG(e);                                   \
  }

IPEX_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CAST)
#undef DEFINE_CAST

} // namespace dpcpp
} // namespace xpu
