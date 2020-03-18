#include <core/Stream.h>
#include <core/Memory.h>


namespace at {
namespace dpcpp {

static void memcpyHostToDevice(void *dst, const void *src, size_t n_bytes, bool async, DP::queue &dpcpp_queue) {
  static const auto write_mode = DP::access::mode::discard_write;
  if (n_bytes == 0)
    return;

  auto cgf = DP_Q_CGF(cgh) {
    auto dst_acc = DPCPPAccessor<write_mode>(cgh, dst, n_bytes).get_access();
    cgh.copy(src, dst_acc);
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(dpcpp_queue, cgf);
  }
}

static void memcpyDeviceToHost(void *dst, const void *src, size_t n_bytes, bool async, DP::queue &dpcpp_queue) {
  static const auto read_mode = DP::access::mode::read;
  if (n_bytes == 0)
    return;

  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc = DPCPPAccessor<read_mode>(cgh, src, n_bytes).get_access();
    cgh.copy(src_acc, dst);
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(dpcpp_queue, cgf);
  }
}

static void memcpyDeviceToDevice(void *dst, const void *src, size_t n_bytes, bool async, DP::queue &dpcpp_queue) {
  static const auto read_mode = DP::access::mode::read;
  static const auto write_mode = DP::access::mode::discard_write;
  if (n_bytes == 0)
    return;
  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc = DPCPPAccessor<read_mode>(cgh, src, n_bytes).get_access();
    auto dst_acc = DPCPPAccessor<write_mode>(cgh, dst, n_bytes).get_access();
    cgh.copy(src_acc, dst_acc);
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(dpcpp_queue, cgf);
  }
}

static void memsetDevice(void* dst, int value, size_t n_bytes, bool async, DP::queue &dpcpp_queue) {
  static const auto write_mode = DP::access::mode::write;

  auto cgf = DP_Q_CGF(cgh) {
    auto dst_acc = DPCPPAccessor<write_mode>(cgh, dst, n_bytes).get_access();
    cgh.fill(dst_acc, (static_cast<uint8_t>(value)));
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(dpcpp_queue, cgf);
  }
}


void dpcppMemcpy(void *dst, const void *src, size_t n_bytes, dpcppMemcpyKind kind) {
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

void dpcppMemcpyAsync(void *dst, const void *src, size_t n_bytes, dpcppMemcpyKind kind) {
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

void dpcppMemset(void *data, int value, size_t n_bytes) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  memsetDevice(data, value, n_bytes, false, dpcpp_queue);
}

void dpcppMemsetAsync(void *data, int value, size_t n_bytes) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  memsetDevice(data, value, n_bytes, true, dpcpp_queue);
}

void* dpcppMalloc(size_t n_bytes) {
  auto ptr = DPCPPmalloc(n_bytes, dpcppGetBufferMap());
  return static_cast<void*>(ptr);
}

void dpcppFree(void* ptr) {
  DPCPPfree(ptr, dpcppGetBufferMap());
}

void dpcppFreeAll() {
  DPCPPfreeAll(dpcppGetBufferMap());
}

} //namespace of dpcpp
} // namespace of c10
