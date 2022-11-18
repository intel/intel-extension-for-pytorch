#include <aten/operators/comm/ScalarType.h>
#include <core/Memory.h>
#include <runtime/Memory.h>

namespace xpu {
namespace dpcpp {

void dpcppMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind) {
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, false);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, false);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, false);
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
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, true);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, true);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, true);
      break;
    default:
      throw std::runtime_error("Unknown dpcpp memory kind");
  }
}

void dpcppMemset(void* data, int value, size_t n_bytes) {
  memsetDevice(data, value, n_bytes, false);
}

void dpcppMemsetAsync(void* data, int value, size_t n_bytes) {
  memsetDevice(data, value, n_bytes, true);
}

template <class T>
void dpcppFill(T* data, T value, size_t n_elems) {
  fillDevice(data, value, n_elems, false);
}

template <class T>
void dpcppFillAsync(T* data, T value, size_t n_elems) {
  fillDevice(data, value, n_elems, true);
}

#define DEFINE_CAST(T, name)                              \
  template <>                                             \
  void dpcppFill(T* data, T value, size_t n_elems) {      \
    fillDevice(data, value, n_elems, false);              \
  }                                                       \
                                                          \
  template <>                                             \
  void dpcppFillAsync(T* data, T value, size_t n_elems) { \
    fillDevice(data, value, n_elems, true);               \
  }

IPEX_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CAST)
#undef DEFINE_CAST

} // namespace dpcpp
} // namespace xpu
