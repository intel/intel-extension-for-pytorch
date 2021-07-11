#include <c10/util/Exception.h>
#include <utils/Macros.h>
#include <runtime/Context.h>
#include <runtime/CachingHostAllocator.h>

#include <deque>
#include <mutex>
#include <unordered_map>
#include <set>

namespace xpu {
namespace dpcpp {

void* CachingHostAllocator::Block::getPtr() const {
  return mPtr;
}

bool CachingHostAllocator::BlockState::hasEvent() {
  return !mEvents.empty();
}

void CachingHostAllocator::BlockState::insertEvent(DPCPP::event& e) {
  mEvents.emplace_back(e);
}

void CachingHostAllocator::BlockState::processEvents() {
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

bool CachingHostAllocator::BlockState::isAllocated() {
  return mAllocated;
}

void CachingHostAllocator::BlockState::setAllocated(bool alloc) {
  mAllocated = alloc;
}

CachingHostAllocator::CachingHostAllocator()
  : mAvailable(Block::Comparator) {}

CachingHostAllocator::~CachingHostAllocator() {
  emptyCache();
}

void CachingHostAllocator::processEvents() {
  for (auto& mb : mBlocks) {
    auto& block_state = mb.second;
    block_state.processEvents();
    if (!block_state.isAllocated() && !block_state.hasEvent()) {
      mAvailable.insert(block_state);
    }
  }
}

bool CachingHostAllocator::isHostPtr(void* ptr) {
  return DPCPP::usm::alloc::host ==
    DPCPP::get_pointer_type(ptr, getDeviceContext());
}

void CachingHostAllocator::emptyCache() {
  std::lock_guard<std::mutex> lock(mMutex);
  processEvents();

  for (auto& blk : mAvailable) {
    auto it = mBlocks.find(blk.getPtr());
    AT_ASSERT(it != mBlocks.end() && !it->second.isAllocated());
    DPCPP::free(blk.getPtr(), getDeviceContext());
    mBlocks.erase(it);
  }

  mAvailable.clear();
}

void CachingHostAllocator::recordEvent(void* ptr, DPCPP::event& e) {
  std::lock_guard<std::mutex> lock(mMutex);

  auto it = mBlocks.find(ptr);
  if (it == mBlocks.end()) {
    return;
  }

  auto& block = it->second;
  block.insertEvent(e);
}

int CachingHostAllocator::malloc(void** ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mMutex);
  processEvents();

  *ptr = nullptr;
  if (size <= 0) {
    return DPCPP_SUCCESS;
  }

  Block block_search(size);
  auto it = mAvailable.lower_bound(block_search);
  if (it != mAvailable.end()) {
    auto& block = mBlocks.at(it->getPtr());
    AT_ASSERT(!block.isAllocated() && !block.hasEvent());
    block.setAllocated(true);
    *ptr = it->getPtr();
    mAvailable.erase(it);
    return DPCPP_SUCCESS;
  }

  *ptr = DPCPP::malloc_host(size, getDeviceContext());
  mBlocks.insert({*ptr, {size, *ptr, true}});
  return DPCPP_SUCCESS;
}

void CachingHostAllocator::release(void* ptr) {
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

}} // namespace xpu::dpcpp
