#include <c10/core/Allocator.h>
#include <core/DPCPPUtils.h>
#include <core/Exception.h>
#include <core/Memory.h>
#include <mutex>

using namespace at::dpcpp;

struct NaiveAllocator {
  // lock around all operations
  std::mutex mutex_;

  void* malloc(size_t num_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ptr = dpcppMalloc(num_bytes);
    return static_cast<void*>(ptr);
  }

  void free(void* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    dpcppFree(p);
  }

  void free_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    dpcppFreeAll();
  }
};

NaiveAllocator naive_allocator;

static inline void NaiveAllocatorDeleter(void* ptr) {
  naive_allocator.free(ptr);
}

struct DPCPPDefaultAllocator : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    at::DeviceIndex device;
    AT_DPCPP_CHECK(dpcppGetDevice(&device));
    void* p = nullptr;
    if (size != 0) {
      p = naive_allocator.malloc(size);
    }
    return {p,
            p,
            &NaiveAllocatorDeleter,
            at::Device(at::DeviceType::DPCPP, device)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &NaiveAllocatorDeleter;
  }
};

DPCPPDefaultAllocator device_allocator;
at::Allocator* DPCPPAllocator_get(void) {
  return &device_allocator;
}
