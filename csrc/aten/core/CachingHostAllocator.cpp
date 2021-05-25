#include <c10/util/Exception.h>
#include <core/Macros.h>
#include <core/Context.h>
#include <core/CachingHostAllocator.h>

#include <deque>
#include <mutex>
#include <unordered_map>
#include <set>


namespace xpu {
namespace dpcpp {

class CHABlock {
public:
  CHABlock(size_t size, void* ptr = nullptr) : mSize(size), mPtr(ptr) {}

  static bool Comparator(const CHABlock& ablock, const CHABlock& bblock) {
    if (ablock.mSize != bblock.mSize) {
      return ablock.mSize < bblock.mSize;
    }
    return (uintptr_t)ablock.mPtr < (uintptr_t)bblock.mPtr;
  }

  void* getPtr() const {
    return mPtr;
  }

private:
  size_t  mSize;
  void*   mPtr;
};

class CHABlockState : public CHABlock {
public:
  CHABlockState(size_t size, void* ptr, bool allocated = false)
    : CHABlock(size, ptr), mAllocated(allocated), mEvents() {}

  bool hasEvent() {
    return !mEvents.empty();
  }

  void insertEvent(DPCPP::event& e) {
    mEvents.emplace_back(e);
  }

  void processEvents() {
    while (hasEvent()) {
      auto& e = mEvents.front();
      bool completed = e.get_info<DPCPP::info::event::command_execution_status>()
        == DPCPP::info::event_command_status::complete;
      if (!completed) {
        return;
      }
      mEvents.pop_front();
    }
  }

  bool isAllocated() {
    return mAllocated;
  }

  void setAllocated(bool alloc) {
    mAllocated = alloc;
  }

private:
  bool    mAllocated;
  std::deque<DPCPP::event> mEvents;
};

class CachingHostAllocator final : public at::Allocator {
public:
  static CachingHostAllocator* Instance() {
    static CachingHostAllocator myInstance;
    return &myInstance;
  }

  static void deleter(void* ptr) {
    Instance()->release(ptr);
  }

  at::DataPtr allocate(size_t size) const override {
    void *ptr = nullptr;
    Instance()->malloc(&ptr, size);
    return {ptr, ptr, &deleter, at::DeviceType::CPU};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &deleter;
  }

  bool isHostPtr(void* ptr) {
    return DPCPP::usm::alloc::host ==
      DPCPP::get_pointer_type(ptr, xpu::dpcpp::getDeviceContext());
  }

  void emptyCache() {
    std::lock_guard<std::mutex> lock(mMutex);
    processEvents();

    for (auto& blk : mAvailable) {
      auto it = mBlocks.find(blk.getPtr());
      AT_ASSERT(it != mBlocks.end() && !it->second.isAllocated());
      DPCPP::free(blk.getPtr(), xpu::dpcpp::getDeviceContext());
      mBlocks.erase(it);
    }

    mAvailable.clear();
  }

  void recordEvent(void* ptr, DPCPP::event& e) {
    std::lock_guard<std::mutex> lock(mMutex);

    auto it = mBlocks.find(ptr);
    if (it == mBlocks.end()) {
      return;
    }

    auto& block = it->second;
    block.insertEvent(e);
  }

  int malloc(void** ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mMutex);
    processEvents();

    *ptr = nullptr;
    if (size <= 0) {
      return DPCPP_SUCCESS;
    }

    CHABlock block_search(size);
    auto it = mAvailable.lower_bound(block_search);
    if (it != mAvailable.end()) {
      auto& block = mBlocks.at(it->getPtr());
      AT_ASSERT(!block.isAllocated() && !block.hasEvent());
      block.setAllocated(true);
      *ptr = it->getPtr();
      mAvailable.erase(it);
      return DPCPP_SUCCESS;
    }

    *ptr = DPCPP::malloc_host(size, xpu::dpcpp::getDeviceContext());
    mBlocks.insert({*ptr, {size, *ptr, true}});
    return DPCPP_SUCCESS;
  }

  void release(void* ptr) {
    std::lock_guard<std::mutex> lock(mMutex);

    if (ptr == nullptr) {
      return;
    }

    auto it = mBlocks.find(ptr);
    AT_ASSERT(it != mBlocks.end());

    auto& block = it->second;
    AT_ASSERT(block.isAllocated());

    block.setAllocated(false);
    processEvents();
  }

private:
  CachingHostAllocator() : mAvailable(CHABlock::Comparator) {}

  void processEvents() {
    for (auto& mb : mBlocks) {
      auto& block = mb.second;
      block.processEvents();
      if (!block.isAllocated() && !block.hasEvent()) {
        mAvailable.insert(block);
      }
    }
  }

  std::mutex mMutex;
  std::unordered_map<void*, CHABlockState> mBlocks;
  std::set<CHABlock, decltype(CHABlock::Comparator)*> mAvailable;
};

// Provide a caching allocator for host allocation by USM malloc_host
Allocator* dpcpp_getCachingHostAllocator() {
  return CachingHostAllocator::Instance();
}

// Record the event on queue where the host allocation is using
void dpcpp_recordEventInCachingHostAllocator(void* ptr, DPCPP::event& e) {
  CachingHostAllocator::Instance()->recordEvent(ptr, e);
}

// Releases all cached host memory allocations
void dpcpp_emptyCacheInCachingHostAllocator() {
  CachingHostAllocator::Instance()->emptyCache();
}

bool dpcpp_isAllocatedByCachingHostAllocator(void* ptr) {
  return CachingHostAllocator::Instance()->isHostPtr(ptr);
}

}} // namespace xpu::dpcpp
