#pragma once

#include <ATen/xpu/XPUContext.h>
#include <utils/DPCPP.h>

#include <core/AllocationInfo.h>
#include <runtime/Device.h>
#include <runtime/XPUGraph.h>

#include <c10/util/flat_hash_map.h>

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

namespace torch_ipex::xpu {
namespace dpcpp {

class CachingDeviceAllocator final {
 private:
  enum class PoolType {
    UNDEF = 0,
    LARGE_POOL = 1,
    SMALL_POOL = 2,
  };

  struct PrivatePool;

  struct Block {
    Block(DeviceId device, sycl::queue queue, size_t size);

    Block(
        DeviceId device,
        sycl::queue queue,
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
        auto a_hash = std::hash<sycl::queue>{}(a->m_queue);
        auto b_hash = std::hash<sycl::queue>{}(b->m_queue);
        return a_hash < b_hash;
      }

      if (a->m_size != b->m_size) {
        return a->m_size < b->m_size;
      }
      return (uintptr_t)a->m_buffer < (uintptr_t)b->m_buffer;
    }

    DeviceId m_device;
    sycl::queue m_queue;
    std::unordered_set<sycl::queue> m_queue_uses;
    size_t m_size;
    PoolType m_pool_type;
    void* m_buffer;
    bool m_allocated;
    Block* m_prev;
    Block* m_next;
    int m_event_cnt;
    // Store pointer to private pool for tracing back
    PrivatePool* m_owner_private_pool;
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

  // Members specific to XPU graphs

  // A recorded graph should retain its memory pool in order all graphs to
  // replay many times on the same active zone, which should not be freed
  // or replaced or modified by other tensors.
  //
  // To identify the pools used only for graphs and should be kept unless
  // no graphs related remained, the graph mechanism has MempoolId_t to
  // mark each pool either created by user or by other graphs. But our
  // allocator is global for all devices but ids are not unique across
  // devices, so there is a need to combine DeviceId together in a key.
  struct PrivatePool {
    PrivatePool()
        : use_count(1),
          large_blocks(Block::Comparator),
          small_blocks(Block::Comparator) {}
    PrivatePool(const PrivatePool&) = delete;
    PrivatePool(PrivatePool&&) = delete;
    PrivatePool& operator=(const PrivatePool&) = delete;
    // Number of live graphs using this pool. When use_count
    // equals to 0, this pool can be destroyed safely.
    // Because SYCL doesn't has the ability to unmap blocks instead of
    // freeing them immediately, there is no need to count remained
    // blocks here as all of them should be considered to be freed once
    // no graph would use this pool anymore.
    int use_count;
    // Totally a mirror copy of a normal block pool, and will always be
    // initialized as empty set when newly create a PrivatePool instance.
    BlockPool large_blocks;
    BlockPool small_blocks;
  };

  struct MempoolHash {
    std::size_t operator()(const std::pair<DeviceId, MempoolId_t>& p) const {
      auto h1 = std::hash<DeviceId>{}(p.first);
      auto h2 = std::hash<CaptureId_t>{}(p.second.first);
      auto h3 = std::hash<CaptureId_t>{}(p.second.second);
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  // Private pools for XPU graphs
  // As DeviceCachingAllocator in IPEX is designed as an singleton running on
  // multi-devices, which is different to the allocator upstream to PyTorch,
  // there is a must to add `DeviceId` into the maping keys list.
  ska::flat_hash_map<
      std::pair<DeviceId, MempoolId_t>,
      std::unique_ptr<PrivatePool>,
      MempoolHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. The reason to use map here is the need to erase PrivatePools
  // in graph_pools at the same time with same search keys.
  ska::
      flat_hash_map<std::pair<DeviceId, MempoolId_t>, PrivatePool*, MempoolHash>
          graph_pools_freeable;
  // Store pools underway in recording.
  std::vector<std::pair<
      std::pair<DeviceId, MempoolId_t>,
      std::function<bool(sycl::queue*)>>>
      recordings_underway;

  MempoolId_t get_mempool_id(DeviceId device);

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

  void find_cached_blocks_bound(
      DeviceId di,
      BlockPool& pool,
      BlockPool::iterator& begin,
      BlockPool::iterator& end);

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
  CachingDeviceAllocator();

  ~CachingDeviceAllocator();

  static CachingDeviceAllocator* Instance(); // Singleton

  std::mutex* getDPCPPFreeMutex() const;

  void malloc(void** devPtr, size_t size, sycl::queue* queue);

  void free(void* buffer);

  void* getBaseAllocation(void* buffer, size_t* outSize);

  void recordQueue(void* buffer, sycl::queue* queue);

  void emptyCache();

  void cacheInfo(DeviceId di, size_t* total, size_t* largest);

  DeviceStats getStatsForDevice(DeviceId di);

  void resetAccumulatedStats(DeviceId di);

  void resetPeakStats(DeviceId di);

  std::vector<SegmentInfo> snapshot() const;

  void dumpMemoryStatus(DeviceId deviceIndex);

  void beginAllocateToPool(
      DeviceId deviceIndex,
      MempoolId_t mempoolId,
      std::function<bool(sycl::queue*)> filter);

  void endAllocateToPool(DeviceId deviceIndex, MempoolId_t mempoolId);

  void releasePool(DeviceId deviceIndex, MempoolId_t mempoolId);
};

} // namespace dpcpp
} // namespace torch_ipex::xpu
