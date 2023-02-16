#ifndef DEVICE_MEMORY_H_
#define DEVICE_MEMORY_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <vector>

// In IPEX, device memory are allocated and reused, and released at last.
// here is a very simple mock for this behavior.

struct DeviceMemoryInfo {
  float* data; // float is enough for the demo
  size_t count;
  bool used;
};

class DeviceMemoryManager {
 public:
  DeviceMemoryManager() {}
  ~DeviceMemoryManager() {}
  void init(sycl::queue* q) {
    this->q = q;
  }
  void deinit();
  float* alloc(size_t count);
  void free(float* data);

 private:
  std::vector<DeviceMemoryInfo> memInfos;
  sycl::queue* q;
};

extern DeviceMemoryManager g_devMemMgr;

#endif