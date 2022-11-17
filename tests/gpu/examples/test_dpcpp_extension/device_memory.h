#ifndef DEVICE_MEMORY_H_
#define DEVICE_MEMORY_H_

#include <CL/sycl.hpp>
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