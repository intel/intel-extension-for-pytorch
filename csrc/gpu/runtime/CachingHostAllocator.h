#pragma once

#include <ATen/Context.h>
#include <ATen/core/ATenGeneral.h>
#include <runtime/Device.h>
#include <utils/DPCPP.h>

namespace xpu {
namespace dpcpp {

class CachingHostAllocator final {
 private:
  class Block {
   public:
    Block(DeviceId device, size_t size, void* ptr = nullptr)
        : mDevId(device), mSize(size), mPtr(ptr) {}

    static bool Comparator(const Block& ablock, const Block& bblock) {
      // If USE_MULTI_CONTEXT=OFF, all the devices share a default context. So
      // there is no need to use device id to represent the same context.
#if defined(USE_MULTI_CONTEXT)
      if (ablock.mDevId != bblock.mDevId) {
        return ablock.mDevId < bblock.mDevId;
      }
#endif
      if (ablock.mSize != bblock.mSize) {
        return ablock.mSize < bblock.mSize;
      }
      return (uintptr_t)ablock.mPtr < (uintptr_t)bblock.mPtr;
    }

    void* getPtr() const;
    sycl::context& getContext() const;

   private:
    // To ensure correct behavior, CachingHostAllocator's deconstructor must be
    // called to free the allocated memory, which is accessible on the host and
    // devices contained in the specified context. In SYCL language, we can use
    // sycl::free to free these memories to avoid memory leaks.
    // To guarantee the same context is used when memory is allocated
    // and deallocated, we need to record the specified context used by
    // sycl::aligned_alloc_host.
    // Fortunately, we can use device id to retrieve the specified context:
    //   1) USE_MULTI_CONTEXT is OFF, all the devices share a default context.
    // we can find the specified context via any device id.
    //   2) USE_MULTI_CONTEXT is ON, in our design, only one context exists per
    // one device. So we can find the specified context via the corresponding
    // device id.
    // For code readability and maintainability, we decide to use device id,
    // which is contained in DeviceGuard's lifetime scope, to represent the
    // specified context.
    DeviceId mDevId; // used to represent sycl::context
    size_t mSize;
    void* mPtr;
  };

  class BlockState : public Block {
   public:
    BlockState(DeviceId device, size_t size, void* ptr, bool allocated = false)
        : Block(device, size, ptr), mAllocated(allocated), mEvents() {}

    bool hasEvent();

    void insertEvent(sycl::event& e);

    void processEvents();

    bool isAllocated();

    void setAllocated(bool alloc);

   private:
    bool mAllocated;
    std::deque<sycl::event> mEvents;
  };

  CachingHostAllocator();

  ~CachingHostAllocator();

  void processEvents();

  std::mutex mMutex;
  std::unordered_map<void*, BlockState> mBlocks;
  std::set<Block, decltype(Block::Comparator)*> mAvailable;

 public:
  static CachingHostAllocator* Instance(); // Singleton

  bool isHostPtr(const void* ptr);

  void emptyCache();

  void recordEvent(void* ptr, sycl::event& e);

  int malloc(void** ptr, size_t size);

  void release(void* ptr);
};

} // namespace dpcpp
} // namespace xpu
