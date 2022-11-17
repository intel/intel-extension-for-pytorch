#include "device_memory.h"

DeviceMemoryManager g_devMemMgr;

void DeviceMemoryManager::deinit() {
  for (auto& info : memInfos) {
    sycl::free(info.data, *q);
  }
}

float* DeviceMemoryManager::alloc(size_t count) {
  if (count == 0) {
    return nullptr;
  }

  for (auto& info : memInfos) {
    if (info.count >= count && !info.used) {
      info.used = true;
#ifdef USE_HOST_MEMORY
      memset(info.data, 0xAB, info.count * sizeof(float));
#endif
      return info.data;
    }
  }

#ifdef USE_HOST_MEMORY
  float* p = sycl::malloc_host<float>(count, *q);
  memset(p, 0xCD, count * sizeof(float));
#else
  float* p = sycl::malloc_device<float>(count, *q);
#endif
  DeviceMemoryInfo info;
  info.data = p;
  info.count = count;
  info.used = true;
  memInfos.push_back(info);

  return p;
}

void DeviceMemoryManager::free(float* data) {
  for (auto& info : memInfos) {
    if (info.data == data) {
      info.used = false;
      return;
    }
  }

  assert(!"should not reach here");
}