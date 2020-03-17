#include <core/Stream.h>
#include <core/Memory.h>


namespace c10 {
namespace sycl {

static void memcpyHostToDevice(void *dst, const void *src, size_t n_bytes, bool async, cl::sycl::queue &sycl_queue) {
  static const auto write_mode = cl::sycl::access::mode::discard_write;
  if (n_bytes == 0)
    return;

  auto cgf = DP_Q_CGF(cgh) {
    auto dst_acc = SYCLAccessor<write_mode>(cgh, dst, n_bytes).get_access();
    cgh.copy(src, dst_acc);
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(sycl_queue, cgf);
  }
}

static void memcpyDeviceToHost(void *dst, const void *src, size_t n_bytes, bool async, cl::sycl::queue &sycl_queue) {
  static const auto read_mode = cl::sycl::access::mode::read;
  if (n_bytes == 0)
    return;

  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc = SYCLAccessor<read_mode>(cgh, src, n_bytes).get_access();
    cgh.copy(src_acc, dst);
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(sycl_queue, cgf);
  }
}

static void memcpyDeviceToDevice(void *dst, const void *src, size_t n_bytes, bool async, cl::sycl::queue &sycl_queue) {
  static const auto read_mode = cl::sycl::access::mode::read;
  static const auto write_mode = cl::sycl::access::mode::discard_write;
  if (n_bytes == 0)
    return;
  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc = SYCLAccessor<read_mode>(cgh, src, n_bytes).get_access();
    auto dst_acc = SYCLAccessor<write_mode>(cgh, dst, n_bytes).get_access();
    cgh.copy(src_acc, dst_acc);
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(sycl_queue, cgf);
  }
}

static void memsetDevice(void* dst, int value, size_t n_bytes, bool async, cl::sycl::queue &sycl_queue) {
  static const auto write_mode = cl::sycl::access::mode::write;

  auto cgf = DP_Q_CGF(cgh) {
    auto dst_acc = SYCLAccessor<write_mode>(cgh, dst, n_bytes).get_access();
    cgh.fill(dst_acc, (static_cast<uint8_t>(value)));
  };

  //launch kernel
  if (async) {
    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
  } else {
    DP_Q_SYNC_SUBMIT(sycl_queue, cgf);
  }
}


void syclMemcpy(void *dst, const void *src, size_t n_bytes, syclMemcpyKind kind) {
  auto& sycl_queue = getCurrentSYCLStream().sycl_queue();
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, false, sycl_queue);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, false, sycl_queue);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, false, sycl_queue);
      break;
    default:
      throw std::runtime_error("Unknown sycl memory kind");
  }
}

void syclMemcpyAsync(void *dst, const void *src, size_t n_bytes, syclMemcpyKind kind) {
  auto& sycl_queue = getCurrentSYCLStream().sycl_queue();
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, true, sycl_queue);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, true, sycl_queue);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, true, sycl_queue);
      break;
    default:
      throw std::runtime_error("Unknown sycl memory kind");
  }
}

void syclMemset(void *data, int value, size_t n_bytes) {
  auto& sycl_queue = getCurrentSYCLStream().sycl_queue();
  memsetDevice(data, value, n_bytes, false, sycl_queue);
}

void syclMemsetAsync(void *data, int value, size_t n_bytes) {
  auto& sycl_queue = getCurrentSYCLStream().sycl_queue();
  memsetDevice(data, value, n_bytes, true, sycl_queue);
}

void* syclMalloc(size_t n_bytes) {
  auto ptr = SYCLmalloc(n_bytes, syclGetBufferMap());
  return static_cast<void*>(ptr);
}

void syclFree(void* ptr) {
  SYCLfree(ptr, syclGetBufferMap());
}

void syclFreeAll() {
  SYCLfreeAll(syclGetBufferMap());
}

} //namespace of sycl
} // namespace of c10
