#pragma once

#include <runtime/Device.h>
#include <runtime/Queue.h>
#include <utils/DPCPP.h>

#include <core/AllocationInfo.h>

#include <algorithm>
#include <bitset>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xpu {
namespace dpcpp {

class CachingDeviceAllocator final {
 private:
  enum class PoolType {
    UNDEF = 0,
    LARGE_POOL = 1,
    SMALL_POOL = 2,
  };

  struct Block {
    Block(DeviceId device, Queue* queue, size_t size);

    Block(
        DeviceId device,
        Queue* queue,
        size_t size,
        PoolType pool_type,
        void* buffer);

    bool is_split() const;

    bool should_split(size_t size);

    static bool Comparator(const Block* a, const Block* b) {
      if (a->m_device != b->m_device) {
        return a->m_device < b->m_device;
      }
      if (a->m_queue != b->m_queue) {
        return (uintptr_t)a->m_queue < (uintptr_t)b->m_queue;
      }
      if (a->m_size != b->m_size) {
        return a->m_size < b->m_size;
      }
      return (uintptr_t)a->m_buffer < (uintptr_t)b->m_buffer;
    }

    DeviceId m_device;
    Queue* m_queue;
    std::unordered_set<Queue*> m_queue_uses;
    size_t m_size;
    PoolType m_pool_type;
    void* m_buffer;
    bool m_allocated;
    Block* m_prev;
    Block* m_next;
    int m_event_cnt;
  };

  using BlockPool = std::set<Block*, decltype(Block::Comparator)*>;
  using StatTypes = std::bitset<static_cast<size_t>(StatType::NUM_TYPES)>;

  mutable std::recursive_mutex mutex;
  mutable std::mutex free_mutex;
  std::vector<DeviceStats> device_stats;
  BlockPool large_blocks;
  BlockPool small_blocks;
  std::unordered_map<void*, Block*> allocated_blocks;
  std::deque<std::pair<sycl::event, Block*>> dpcpp_events;

  CachingDeviceAllocator();

  ~CachingDeviceAllocator();

  DeviceStats& get_stats_for_device(DeviceId device);

  void update_stat_array(
      StatArray& stat_array,
      int64_t amount,
      const StatTypes& stat_types);

  int malloc_with_retry(DeviceId di, void** devPtr, size_t size);

  std::vector<const Block*> get_all_blocks() const;

  void free_block(Block* block);

  void free_blocks(
      BlockPool& blocks,
      BlockPool::iterator it,
      BlockPool::iterator end);

  void free_cached_blocks(DeviceId di);

  size_t try_merge_blocks(Block* dst, Block* src, BlockPool* pool);

  StatType get_stat_type_for_pool(const PoolType pool_type);

  Block* find_allocated_block(void* buffer);

  void insert_events(Block* block);

  void process_events();

  void synchronize_and_free_events(std::optional<DeviceId> di);

  void cache_info_aux(
      BlockPool& blocks,
      DeviceId di,
      size_t* total,
      size_t* largest);

 public:
  static CachingDeviceAllocator* Instance(); // Singleton

  std::mutex* getDPCPPFreeMutex() const;

  void malloc(void** devPtr, size_t size, Queue* queue);

  void free(void* buffer);

  void* getBaseAllocation(void* buffer, size_t* outSize);

  void recordQueue(void* buffer, Queue* queue);

  void emptyCache();

  void cacheInfo(DeviceId di, size_t* total, size_t* largest);

  DeviceStats getStatsForDevice(DeviceId di);

  void resetAccumulatedStats(DeviceId di);

  void resetPeakStats(DeviceId di);

  std::vector<SegmentInfo> snapshot() const;

  void dumpMemoryStatus(DeviceId deviceIndex);
};

} // namespace dpcpp
} // namespace xpu
