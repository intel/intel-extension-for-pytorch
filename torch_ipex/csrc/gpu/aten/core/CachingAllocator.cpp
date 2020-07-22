#include <core/CachingAllocator.h>
#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <c10/core/Allocator.h>
#include <tensor/Context.h>

#include <optional>
#include <algorithm>
#include <bitset>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>


// C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace at {
namespace dpcpp {

//
// Caching allocator for dpcpp device memory allocations.
//
// - Allocations are associated with a queue. Once freed, blocks can be
//   re-allocated on the same queue, but not on any other queue.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to dpcppMalloc.
// - If the dpcppMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using dpcppMalloc.
//   To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with queues, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple queues.
//
// The library provides a recordQueue() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded queue completes
// work.
//


using queue_set = std::unordered_set<at::dpcpp::DPCPPStream>;

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocs to 2 MiB

typedef std::bitset<static_cast<size_t>(at::dpcpp::CAStatType::NUM_TYPES)> CAStatTypes;

void update_stat(at::dpcpp::CAStat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT(stat.current >= 0, "Negative tracked stat in DPCPP Caching allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(at::dpcpp::CAStat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(at::dpcpp::CAStat& stat) {
  stat.peak = stat.current;
}

void update_stat_array(at::dpcpp::CAStatArray& stat_array, int64_t amount, const at::dpcpp::CAStatTypes& stat_types) {
  for (size_t stat_type = 0; stat_type < stat_types.size(); ++stat_type) {
    if (stat_types[stat_type]) {
      update_stat(stat_array[stat_type], amount);
    }
  }
}

struct CABlock;
typedef bool (*Comparison)(const CABlock*, const CABlock*);
typedef std::set<CABlock*, Comparison> CABlockPool;

struct CABlock {
  at::DeviceIndex   device;      // gpu
  DPCPP::queue* queuePtr;    // allocation queue
  queue_set     queue_uses;  // queues on which the block was used
  size_t        size;        // block size in bytes
  CABlockPool*  pool;        // owning memory pool
  void*         ptr;         // memory address
  bool          allocated;   // in-use flag
  CABlock*      prev;        // prev block if split from a larger allocation
  CABlock*      next;        // next block if split from a larger allocation
  int           event_count; // number of outstanding DPCPP events

  CABlock(at::DeviceIndex device,  DPCPP::queue *pQueue, size_t size, CABlockPool* pool, void* ptr) :
    device(device), queuePtr(pQueue), queue_uses(), size(size), pool(pool),
    ptr(ptr), allocated(false), prev(nullptr), next(nullptr), event_count(0) { }

  // constructor for search key
  CABlock(at::DeviceIndex device, DPCPP::queue *pQueue, size_t size) :
    device(device), queuePtr(pQueue), queue_uses(), size(size), pool(nullptr),
    ptr(nullptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

static bool CABlockComparator(const CABlock* a, const CABlock* b)
{
  if (a->device != b->device) {
    return a->device < b->device;
  }
  if (a->queuePtr != b->queuePtr) {
    return (uintptr_t)a->queuePtr < (uintptr_t)b->queuePtr;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

DPCPP_DEF_K1(CA_dummy_kernel);

class CachingAllocator {

 private:

  // lock around all operations
  mutable std::recursive_mutex mutex;

  // lock around calls to dpcppFree
  mutable std::mutex dpcpp_free_mutex;

  // device statistics
  std::vector<at::dpcpp::CADeviceStats> device_stats;

  // unallocated cached blocks larger than 1 MB
  CABlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  CABlockPool small_blocks;

  // allocated blocks by device pointer
  std::unordered_map<void*, CABlock*> allocated_blocks;

  // outstanding dpcpp events
  std::deque<std::pair<DPCPP::event, CABlock*>> dpcpp_events;

 public:

  CachingAllocator() :
      large_blocks(CABlockComparator),
      small_blocks(CABlockComparator) {}

  std::mutex* getDPCPPFreeMutex() const {
    return &dpcpp_free_mutex;
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, size_t size, DPCPP::queue *queuePtr)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    DeviceIndex curDevID;
		AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));

    // process outstanding dpcppEvents
    process_events();

    size = round_size(size);

    CABlock search_key(curDevID, queuePtr, size);
    auto& pool = get_pool(size);

    CADeviceStats& stats = get_stats_for_device(curDevID);
    CAStatTypes stat_types;
    stat_types[static_cast<size_t>(CAStatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    auto find_free_block = [&]()->CABlock*{
      auto it = pool.lower_bound(&search_key);
      if (it != pool.end() && (*it)->device == curDevID &&
          (*it)->queuePtr == queuePtr) {
        CABlock* block = *it;
        pool.erase(it);
        return block;
      }
      return nullptr;
    };

    CABlock* block = find_free_block();
#if 0
    if (block == nullptr) {
      bool freed_memory = false;
      for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
        freed_memory |=
            FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
      }
      if (freed_memory) {
        block = find_free_block();
      }
    }
#endif
    if (block == nullptr) {
      void* ptr;
      size_t alloc_size = get_allocation_size(size);
      int err = dpcpp_malloc_with_retry(curDevID, &ptr, alloc_size);

      if (err == DPCPP_SUCCESS) {
        block = new CABlock(curDevID, queuePtr, alloc_size, &pool, ptr);
        update_stat_array(stats.segment, 1, stat_types);
        update_stat_array(stats.reserved_bytes, alloc_size, stat_types);
      } else {
				// TODO: Call DPCPP Runtime API to verify no available memory
        auto dpcppDev = dpcppGetRawDevice(curDevID);
        size_t device_total = dpcppDev.get_info<DPCPP::info::device::global_mem_size>();
        stats.num_ooms += 1;
        
        // "total capacity": total global memory on GPU
        // "already allocated": memory allocated by the program using the
        //                      caching allocator
        // "cached": memory held by the allocator but not used by the program
        //
        // The "allocated" amount  does not include memory allocated outside
        // of the caching allocator, such as memory allocated by other programs
        // or memory held by the driver.
        //
        // The sum of "allocated" + "free" + "cached" may be less than the
        // total capacity due to memory held by the driver and usage by other
        // programs.
        //
        // Note that at this point dpcpp_malloc_with_retry has already returned all
        // possible "cached" memory to the driver. The only remaining "cached"
        // memory is split from a larger block that is partially in-use.
        AT_ERROR("DPCPP out of memory. Tried to allocate ", format_size(alloc_size),
          " (GPU ", curDevID, "; ",
          format_size(device_total), " total capacity; ",
          format_size(stats.allocated_bytes[static_cast<size_t>(CAStatType::AGGREGATE)].current),
          " already allocated; ",
          format_size(stats.reserved_bytes[static_cast<size_t>(CAStatType::AGGREGATE)].current),
          " reserved in total by PyTorch)");
      }
    }

    CABlock* remaining = nullptr;
    AT_ASSERT(block);

    const bool already_split = block->is_split();
    if (should_split(block, size)) {
      remaining = block;

      block = new CABlock(curDevID, queuePtr, size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      pool.insert(remaining);

      if (already_split) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(stats.inactive_split_bytes, -block->size, stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit block,
        // size remaining->size bytes.
        update_stat_array(stats.inactive_split_bytes, remaining->size, stat_types);
        update_stat_array(stats.inactive_split, 1, stat_types);
      }
    } else if (already_split) {
      // An already-split block is becoming active
      update_stat_array(stats.inactive_split_bytes, -block->size, stat_types);
      update_stat_array(stats.inactive_split, -1, stat_types);
    }

    block->allocated = true;
    allocated_blocks[block->ptr] = block;

    *devPtr = block->ptr;

    update_stat_array(stats.allocation, 1, stat_types);
    update_stat_array(stats.allocated_bytes, block->size, stat_types);
    update_stat_array(stats.active, 1, stat_types);
    update_stat_array(stats.active_bytes, block->size, stat_types);
  }

  void free(void* ptr)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!ptr) {
      return;
    }

    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      AT_ERROR("invalid device pointer: ", ptr);
    }

    CABlock* block = it->second;
    allocated_blocks.erase(it);
    block->allocated = false;

		CADeviceStats& stats = get_stats_for_device(block->device);
    CAStatTypes stat_types;
    stat_types[static_cast<size_t>(CAStatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    update_stat_array(stats.allocation, -1, {stat_types});
    update_stat_array(stats.allocated_bytes, -block->size, {stat_types});

    if (!block->queue_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    CABlock* block = find_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    while (block->prev) {
      block = block->prev;
    }
    void *basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordQueue(const at::DataPtr& ptr, at::dpcpp::DPCPPStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when DPCPP tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &dpcpp_raw_delete)
      return;

    std::lock_guard<std::recursive_mutex> lock(mutex);

    CABlock* block = find_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
		auto &queue = stream.dpcpp_queue();
    if (&queue == block->queuePtr) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->queue_uses.insert(stream);
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    synchronize_and_free_events(nullopt);
    free_blocks(large_blocks, large_blocks.begin(), large_blocks.end());
    free_blocks(small_blocks, small_blocks.begin(), small_blocks.end());
  }

  /** Retrieves info (total size + largest block) of the memory cache **/
  void cacheInfo(at::DeviceIndex di, size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cache_info_aux(large_blocks, di, total, largest);
    cache_info_aux(small_blocks, di, total, largest);
  }

  /** Returns a copy of the memory allocator stats for the device **/
  at::dpcpp::CADeviceStats getStatsForDevice(at::DeviceIndex di) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return get_stats_for_device(di);
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats(at::DeviceIndex di) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    CADeviceStats& stats = get_stats_for_device(di);

    for (size_t statType = 0; statType < static_cast<size_t>(CAStatType::NUM_TYPES); ++statType) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats(at::DeviceIndex di) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    CADeviceStats& stats = get_stats_for_device(di);

    for (size_t statType = 0; statType < static_cast<size_t>(CAStatType::NUM_TYPES); ++statType) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
    }
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. **/
  std::vector<at::dpcpp::CASegmentInfo> snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<at::dpcpp::CASegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const CABlock* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      CASegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.is_large = (head_block->pool == &large_blocks);

      const CABlock* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        CABlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0);

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
        }

        block = block->next;
      }
    }

    std::sort(result.begin(), result.end(), [](const CASegmentInfo& a, const CASegmentInfo& b) {
      if (a.device != b.device) {
        return a.device < b.device;
      }
      return a.address < b.address;
    });

    return result;
  }

 private:

  // All private methods do not acquire the allocator mutex.

  at::dpcpp::CADeviceStats& get_stats_for_device(at::DeviceIndex device) {
    TORCH_CHECK(device >= 0);
    if ((size_t) device >= device_stats.size()) {
      device_stats.resize(device + 1);
    }
    return device_stats.at(device);
  }

  std::vector<const CABlock*> get_all_blocks() const {
    std::vector<const CABlock*> blocks;
    blocks.insert(blocks.end(), small_blocks.begin(), small_blocks.end());
    blocks.insert(blocks.end(), large_blocks.begin(), large_blocks.end());
    for (const auto& item : allocated_blocks) {
      blocks.push_back(item.second);
    }
    return blocks;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(CABlock* block)
  {
    AT_ASSERT(!block->allocated && block->event_count == 0);

    size_t original_block_size = block->size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<CABlock*, 2> merge_candidates = {block->prev, block->next};
    for (CABlock* merge_candidate : merge_candidates) {
      const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    pool.insert(block);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    CADeviceStats& stats = get_stats_for_device(block->device);
    CAStatTypes stat_types;
    stat_types[static_cast<size_t>(CAStatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, stat_types);
    update_stat_array(stats.active, -1, stat_types);
    update_stat_array(stats.active_bytes, -original_block_size, stat_types);
  }

  /** combine previously split blocks. returns the size of the subsumed block, or 0 on failure. */
  size_t try_merge_blocks(CABlock* dst, CABlock* src, CABlockPool& pool)
  {
    if (!src || src->allocated || src->event_count > 0) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    pool.erase(src);
    delete src;

    return subsumed_size;
  }

  CABlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  at::dpcpp::CAStatType get_stat_type_for_pool(const CABlockPool& pool) {
    if (&pool == &small_blocks) {
      return CAStatType::SMALL_POOL;
    } else if (&pool == &large_blocks) {
      return CAStatType::LARGE_POOL;
    } else {
      AT_ERROR("get_stat_type_for_pool: invalid pool");
    }
  }

  bool should_split(const CABlock* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool == &small_blocks) {
      return remaining >= kMinBlockSize;
    } else if (block->pool == &large_blocks) {
      return remaining > kSmallSize;
    } else {
      AT_ERROR("should_split: invalid pool");
    }
  }

  size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  int dpcpp_malloc_with_retry(DeviceIndex di, void** devPtr, size_t size)
  {
    // Try USM malloc. If USM malloc fails, frees all non-split cached blocks
    // and retries.
		auto syclDev = at::dpcpp::dpcppGetRawDevice(di);
		*devPtr = DPCPP::malloc_device(size, syclDev, at::dpcpp::getGlobalContext());


    if (*devPtr == NULL) {
      CADeviceStats& stats = get_stats_for_device(di);
      stats.num_alloc_retries += 1;
      free_cached_blocks(di);
			*devPtr = DPCPP::malloc_device(size, syclDev, at::dpcpp::getGlobalContext());
      if (*devPtr == NULL) {
        return DPCPP_FAILURE;
      }
    }

    return DPCPP_SUCCESS;
  }

  void free_cached_blocks(DeviceIndex di)
  {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(di);

    // Free all non-split cached blocks on device
    CABlock lower_bound(di, nullptr, 0);
    CABlock upper_bound(di + 1, nullptr, 0);

    free_blocks(
        large_blocks,
        large_blocks.lower_bound(&lower_bound),
        large_blocks.lower_bound(&upper_bound));
    free_blocks(
        small_blocks,
        small_blocks.lower_bound(&lower_bound),
        small_blocks.lower_bound(&upper_bound));
  }

  void free_blocks(CABlockPool& blocks, CABlockPool::iterator it, CABlockPool::iterator end)
  {
    // Frees all non-split blocks between `it` and `end`
    while (it != end) {
      CABlock* block = *it;
      if (!block->prev && !block->next) {
        DPCPP::free((void*)block->ptr, at::dpcpp::getGlobalContext());

        CADeviceStats& stats = get_stats_for_device(block->device);
        CAStatTypes stat_types;
        stat_types[static_cast<size_t>(CAStatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
        update_stat_array(stats.segment, -1, stat_types);
        update_stat_array(stats.reserved_bytes, -block->size, stat_types);

        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }

  void synchronize_and_free_events(optional<DeviceIndex> di) {
    // Synchronize on outstanding events and then free associated blocks.
    // Limited to blocks on the given device if specified.

    auto remaining_events = decltype(dpcpp_events)();

    for (auto& e : dpcpp_events) {
      DPCPP::event event = e.first;
      CABlock* block = e.second;
      if (di.has_value() && block->device != *di) {
        remaining_events.push_back(e);
        continue;
      }
			event.wait();
      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    std::swap(dpcpp_events, remaining_events);
  }

  CABlock* find_allocated_block(void *ptr) {
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    return it->second;
  }

  void insert_events(CABlock* block)
  {
    queue_set queues(std::move(block->queue_uses));
    AT_ASSERT(block->queue_uses.empty());
    for (auto it = queues.begin(); it != queues.end(); ++it) {
      // TODO: following dummy kernel should be replaced by barrier
      auto cgf = DPCPP_Q_CGF(cgh) {
        cgh.single_task<DPCPP_K(CA_dummy_kernel)>([=]() {});
      };
      auto event = it->dpcpp_queue().submit(cgf);
      block->event_count++;
      dpcpp_events.emplace_back(event, block);
    }
  }

  void process_events()
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!dpcpp_events.empty()) {
      auto& e = dpcpp_events.front();
      auto event = e.first;
      CABlock* block = e.second;
      bool event_completed = 
        event.get_info<DPCPP::info::event::command_execution_status>() == 
        DPCPP::info::event_command_status::complete;
      if (!event_completed) {
        break;
      }
      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      dpcpp_events.pop_front();
    }
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cache_info_aux(CABlockPool& blocks, DeviceIndex di, size_t* total, size_t* largest)
  {
    CABlock search_key(di, 0, 0);
    auto it = blocks.lower_bound(&search_key);
    for (; it != blocks.end() && *it && (*it)->device == di; ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }
};

CachingAllocator caching_allocator;

struct DPCPPCachingAllocator : public Allocator {
  DataPtr allocate(size_t size) const override {
    DeviceIndex curDevID;
    AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
    void* r = nullptr;
    if (size != 0) {
      auto& dpcpp_queue = getCurrentDPCPPStream(curDevID).dpcpp_queue();
      caching_allocator.malloc(&r, size, &dpcpp_queue);
    }
    auto ctx = new at::AtenIpexTypeDPCPP::DPCPPTensorContext(r);
    return {r, ctx, &dpcpp_raw_delete, Device(DeviceType::DPCPP, curDevID)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &dpcpp_raw_delete;
  }
};

DPCPPCachingAllocator device_allocator;

Allocator* dpcpp_getCachingAllocator(void)
{
  return &device_allocator;
}

void dpcpp_emptyCache(void) {
  caching_allocator.emptyCache();
}

void dpcpp_cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.cacheInfo(dev_id, cachedAndFree, largestBlock);
}

void* dpcpp_getBaseAllocation(void *ptr, size_t *size)
{
  return caching_allocator.getBaseAllocation(ptr, size);
}

void dpcpp_recordQueue(const DataPtr& ptr, at::dpcpp::DPCPPStream stream) {
  caching_allocator.recordQueue(ptr, stream);
}

std::mutex* getFreeMutex()
{
  return caching_allocator.getDPCPPFreeMutex();
}

static inline void assertValidDevice(DeviceIndex di) {
  int device_num = (int)at::dpcpp::device_count();
  AT_ASSERTM(0 <= (int)di && (int)di < device_num, "Invalid device argument.");
}

CADeviceStats getDeviceStats(DeviceIndex di) {
  assertValidDevice(di);
  return caching_allocator.getStatsForDevice(di);
}

void dpcpp_resetAccumulatedStats(DeviceIndex di) {
  assertValidDevice(di);
  caching_allocator.resetAccumulatedStats(di);
}

void dpcpp_resetPeakStats(DeviceIndex di) {
  assertValidDevice(di);
  caching_allocator.resetPeakStats(di);
}

std::vector<CASegmentInfo> dpcpp_snapshot() {
  return caching_allocator.snapshot();
}

void* dpcpp_raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  void* r = nullptr;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  caching_allocator.malloc(&r, nbytes, &dpcpp_queue);
  return r;
}

void* dpcpp_raw_alloc_with_queue(size_t nbytes, DPCPP::queue &queue) {
  if (nbytes == 0) {
    return nullptr;
  }
  void* r = nullptr;
  caching_allocator.malloc(&r, nbytes, &queue);
  return r;
}

void dpcpp_raw_delete(void* ptr) {
  auto ctx = (at::AtenIpexTypeDPCPP::DPCPPTensorContext*)ptr;
  auto data = ctx->data();
  caching_allocator.free(data);
  delete ctx;
}

}} // namespace aten::dpcpp

