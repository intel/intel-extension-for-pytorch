#include <c10/core/Allocator.h>
#include <c10/core/Allocator.h>
#include <core/DPCPPUtils.h>
#include <core/Exception.h>
#include <core/Memory.h>
#include <tensor/Context.h>
#include <mutex>


using namespace at::dpcpp;
using namespace at::AtenIpexTypeDPCPP;

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
  auto ctx = (DPCPPTensorContext*)ptr;
  auto data = ctx->data();
  naive_allocator.free(data);
  delete ctx;
}

struct DPCPPDefaultAllocator : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    at::DeviceIndex device;
    AT_DPCPP_CHECK(dpcppGetDevice(&device));
    void* p = nullptr;
    if (size != 0) {
      p = naive_allocator.malloc(size);
    }
    auto ctx = new DPCPPTensorContext(p);
    return {p,
            ctx,
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
