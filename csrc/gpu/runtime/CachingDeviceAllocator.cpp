#include <runtime/CachingDeviceAllocator.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>
#include <utils/Helpers.h>

namespace torch_ipex::xpu {
namespace dpcpp {

// NOTE: Make it global to avoid its destruction
// too early during workload exit
static CachingDeviceAllocator myInstance;

constexpr size_t kDevAlignment = 512;
constexpr size_t kMinBlockSize = 512;
constexpr size_t kSmallSize = 1048576;
constexpr size_t kSmallBuffer = 2097152;
constexpr size_t kLargeBuffer = 20971520;
constexpr size_t kMinLargeAlloc = 10485760;
constexpr size_t kRoundLarge = 2097152;
// Single allocation must be less than 4GB with stateful memory management in
// underlying runtime.
constexpr size_t kStatefulLimit = 4294967296;

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

static inline void assertValidDevice(DeviceId di) {
  auto dev_cnt = at::xpu::device_count();
  AT_ASSERTM(0 <= (int)di && (int)di < dev_cnt, "Invalid device argument.");
}

static void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT(
      stat.current >= 0,
      "Negative tracked stat in XPU Caching allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }

  if (amount < 0) {
    stat.freed += -amount;
  }
}

CachingDeviceAllocator* CachingDeviceAllocator::Instance() {
  return &myInstance;
}

CachingDeviceAllocator::~CachingDeviceAllocator() {
  // Do not empty cache here to avoid dependence issue
  // between other components
}

void CachingDeviceAllocator::update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for (size_t stat_type = 0; stat_type < stat_types.size(); ++stat_type) {
    if (stat_types[stat_type]) {
      update_stat(stat_array[stat_type], amount);
    }
  }
}

CachingDeviceAllocator::Block::Block(
    DeviceId device,
    sycl::queue queue,
    size_t size)
    : m_device(device),
      m_queue(queue),
      m_queue_uses(),
      m_size(size),
      m_pool_type(PoolType::UNDEF),
      m_buffer(nullptr),
      m_allocated(0),
      m_prev(nullptr),
      m_next(nullptr),
      m_event_cnt(0),
      m_owner_private_pool(nullptr) {
  auto device_cnt = at::xpu::device_count();
  std::vector<DeviceStats> dev_stats;
}

CachingDeviceAllocator::Block::Block(
    DeviceId device,
    sycl::queue queue,
    size_t size,
    PoolType pool_type,
    void* buffer)
    : m_device(device),
      m_queue(queue),
      m_queue_uses(),
      m_size(size),
      m_pool_type(pool_type),
      m_buffer(buffer),
      m_allocated(false),
      m_prev(nullptr),
      m_next(nullptr),
      m_event_cnt(0),
      m_owner_private_pool(nullptr) {}

bool CachingDeviceAllocator::Block::is_split() const {
  return (m_prev != nullptr) || (m_next != nullptr);
}

bool CachingDeviceAllocator::Block::should_split(size_t size) {
  size_t remaining = m_size - size;
  if (m_pool_type == PoolType::SMALL_POOL) {
    return remaining >= kMinBlockSize;
  } else if (m_pool_type == PoolType::LARGE_POOL) {
    return remaining > kSmallSize;
  } else {
    AT_ERROR("should_split: invalid pool");
  }
}

CachingDeviceAllocator::CachingDeviceAllocator()
    : large_blocks(Block::Comparator), small_blocks(Block::Comparator) {}

std::mutex* CachingDeviceAllocator::getDPCPPFreeMutex() const {
  return &free_mutex;
}

int CachingDeviceAllocator::malloc_with_retry(
    DeviceId di,
    void** devPtr,
    size_t size) {
  auto& syclDev = at::xpu::get_raw_device(di);
  // Our minimum allocated memory is 512. Thus we set mem align to 512.
  *devPtr = sycl::aligned_alloc_device(
      kDevAlignment, size, syclDev, at::xpu::get_device_context());

  if (*devPtr == NULL) {
    DeviceStats& stats = get_stats_for_device(di);
    stats.num_alloc_retries += 1;
    free_cached_blocks(di);
    *devPtr = sycl::aligned_alloc_device(
        kDevAlignment, size, syclDev, at::xpu::get_device_context());
    if (*devPtr == NULL) {
      return DPCPP_FAILURE;
    }
  }

  return DPCPP_SUCCESS;
}

void CachingDeviceAllocator::malloc(
    void** devPtr,
    size_t asize,
    sycl::queue* queue) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  DeviceId curDevID = at::xpu::current_device();
  process_events();

  auto size = (asize < kMinBlockSize)
      ? kMinBlockSize
      : (kMinBlockSize * ((asize + kMinBlockSize - 1) / kMinBlockSize));

  if ((!Settings::I().has_2d_block_array(curDevID)) &&
      (asize > kStatefulLimit)) {
    size_t device_total = dpcppGlobalMemSize(curDevID);
    DeviceStats& stats = get_stats_for_device(curDevID);
    TORCH_CHECK(
        false,
        "Current platform can NOT allocate memory block with size larger than 4GB! Tried to allocate ",
        format_size(asize),
        " (GPU  ",
        (int)curDevID,
        "; ",
        format_size(device_total),
        " total capacity; ",
        format_size(
            stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                .current),
        " already allocated; ",
        format_size(
            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                .current),
        " reserved in total by PyTorch)");
  }

  BlockPool* pool = nullptr;
  PrivatePool* private_pool = nullptr;
  PoolType pool_type = PoolType::UNDEF;
  Block* block = nullptr;

  if (recordings_underway.size()) {
    // graph path, try to find the blocks pointer which related to the
    // PrivatePool who is recording graph on current queue.
    for (auto& entry : recordings_underway) {
      if (entry.second(queue)) {
        auto it1 = graph_pools.find(entry.first);
        TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
        if (size <= kSmallSize) {
          pool_type = PoolType::SMALL_POOL;
          pool = &it1->second->small_blocks;
        } else {
          pool_type = PoolType::LARGE_POOL;
          pool = &it1->second->large_blocks;
        }
        private_pool = it1->second.get();
      }
    }
  }
  // fallback check. It's not suitable to change it to 'else' statement.
  if (pool == nullptr) {
    // normal path, search and return aiming block for allocation in
    // DeviceCachingAllocator's own pool.
    if (size <= kSmallSize) {
      pool_type = PoolType::SMALL_POOL;
      pool = &small_blocks;
    } else {
      pool_type = PoolType::LARGE_POOL;
      pool = &large_blocks;
    }
  }

  Block search_key(curDevID, *queue, size);
  auto find_free_block = [&]() -> Block* {
    auto it = pool->lower_bound(&search_key);
    if (it != pool->end() && (*it)->m_device == curDevID &&
        (*it)->m_queue == *queue) {
      Block* block = *it;
      pool->erase(it);
      return block;
    }
    return nullptr;
  };

  StatTypes stat_types;
  stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
  stat_types[static_cast<size_t>(get_stat_type_for_pool(pool_type))] = true;
  DeviceStats& stats = get_stats_for_device(curDevID);
  block = find_free_block();

  if (block == nullptr) {
    void* buffer;
    size_t alloc_size = (size <= kSmallSize)
        ? kSmallBuffer
        : ((size < kMinLargeAlloc)
               ? kLargeBuffer
               : (kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge)));

    int err = malloc_with_retry(curDevID, &buffer, alloc_size);

    if (err == DPCPP_SUCCESS) {
      block = new Block(curDevID, *queue, alloc_size, pool_type, buffer);
      update_stat_array(stats.segment, 1, stat_types);
      update_stat_array(stats.reserved_bytes, alloc_size, stat_types);
    } else {
      auto& dpcppDev = at::xpu::get_raw_device(curDevID);
      size_t device_total = dpcppGlobalMemSize(curDevID);
      stats.num_ooms += 1;

      TORCH_CHECK(
          false,
          "XPU out of memory. Tried to allocate ",
          format_size(alloc_size),
          " (GPU ",
          (int)curDevID,
          "; ",
          format_size(device_total),
          " total capacity; ",
          format_size(
              stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current),
          " already allocated; ",
          format_size(
              stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current),
          " reserved in total by PyTorch)");
    }
  }

  Block* remaining = nullptr;
  AT_ASSERT(block);

  // need to record the block's owner pool for lazy releasing
  block->m_owner_private_pool = private_pool;

  const bool already_split = block->is_split();
  if (block->should_split(size)) {
    remaining = block;

    block = new Block(curDevID, *queue, size, pool_type, block->m_buffer);
    block->m_prev = remaining->m_prev;
    if (block->m_prev) {
      block->m_prev->m_next = block;
    }
    block->m_next = remaining;

    remaining->m_prev = block;
    remaining->m_buffer = static_cast<char*>(remaining->m_buffer) + size;
    remaining->m_size -= size;
    pool->insert(remaining);

    if (already_split) {
      update_stat_array(stats.inactive_split_bytes, -block->m_size, stat_types);
    } else {
      update_stat_array(
          stats.inactive_split_bytes, remaining->m_size, stat_types);
      update_stat_array(stats.inactive_split, 1, stat_types);
    }
  } else if (already_split) {
    update_stat_array(stats.inactive_split_bytes, -block->m_size, stat_types);
    update_stat_array(stats.inactive_split, -1, stat_types);
  }

  block->m_allocated = true;
  allocated_blocks[block->m_buffer] = block;

  *devPtr = block->m_buffer;

  update_stat_array(stats.allocation, 1, stat_types);
  update_stat_array(stats.allocated_bytes, block->m_size, stat_types);
  update_stat_array(stats.active, 1, stat_types);
  update_stat_array(stats.active_bytes, block->m_size, stat_types);
}

void CachingDeviceAllocator::free(void* buffer) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (!buffer) {
    return;
  }

  auto it = allocated_blocks.find(buffer);
  if (it == allocated_blocks.end()) {
    AT_ERROR("invalid device pointer: ", buffer);
  }

  Block* block = it->second;
  allocated_blocks.erase(it);
  block->m_allocated = false;

  DeviceStats& stats = get_stats_for_device(block->m_device);
  StatTypes stat_types;
  stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
  stat_types[static_cast<size_t>(get_stat_type_for_pool(block->m_pool_type))] =
      true;
  update_stat_array(stats.allocation, -1, {stat_types});
  update_stat_array(stats.allocated_bytes, -block->m_size, {stat_types});

  if (!block->m_queue_uses.empty()) {
    insert_events(block);
  } else {
    free_block(block);
  }
}

void* CachingDeviceAllocator::getBaseAllocation(void* buffer, size_t* outSize) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  Block* block = find_allocated_block(buffer);
  if (!block) {
    AT_ERROR("invalid device pointer: ", buffer);
  }
  while (block->m_prev) {
    block = block->m_prev;
  }
  void* basePtr = block->m_buffer;
  if (outSize) {
    size_t size = 0;
    while (block) {
      size += block->m_size;
      block = block->m_next;
    }
    *outSize = size;
  }
  return basePtr;
}

void CachingDeviceAllocator::recordQueue(void* buffer, sycl::queue* queue) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  Block* block = find_allocated_block(buffer);
  TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
  if (*queue == block->m_queue) {
    return;
  }

  block->m_queue_uses.insert(*queue);
}

void CachingDeviceAllocator::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  synchronize_and_free_events(std::nullopt);

  /*
   * See Note [Safe to Free Blocks on BlockPool]
   *
   * torch.xpu.empty_cache will release all unoccupied cached memory currently
   * held on all the GPUs. So we have to do a device-level synchronization on
   * all GPUs.
   */
  auto count = at::xpu::device_count();
  for (auto i = 0; i < count; i++) {
    at::xpu::syncStreamsOnDevice(i);
  }

  free_blocks(large_blocks, large_blocks.begin(), large_blocks.end());
  free_blocks(small_blocks, small_blocks.begin(), small_blocks.end());
}

StatType CachingDeviceAllocator::get_stat_type_for_pool(
    const PoolType pool_type) {
  if (pool_type == PoolType::SMALL_POOL) {
    return StatType::SMALL_POOL;
  } else if (pool_type == PoolType::LARGE_POOL) {
    return StatType::LARGE_POOL;
  } else {
    AT_ERROR("get_stat_type_for_pool: invalid pool");
  }
}

CachingDeviceAllocator::Block* CachingDeviceAllocator::find_allocated_block(
    void* buffer) {
  auto it = allocated_blocks.find(buffer);
  if (it == allocated_blocks.end()) {
    return nullptr;
  }
  return it->second;
}

void CachingDeviceAllocator::free_block(Block* block) {
  AT_ASSERT(!block->m_allocated && block->m_event_cnt == 0);

  size_t original_block_size = block->m_size;

  BlockPool* pool = nullptr;
  if (block->m_owner_private_pool == nullptr) {
    if (block->m_pool_type == PoolType::LARGE_POOL) {
      pool = &large_blocks;
    } else if (block->m_pool_type == PoolType::SMALL_POOL) {
      pool = &small_blocks;
    }
  } else {
    if (block->m_pool_type == PoolType::LARGE_POOL) {
      pool = &block->m_owner_private_pool->large_blocks;
    } else if (block->m_pool_type == PoolType::SMALL_POOL) {
      pool = &block->m_owner_private_pool->small_blocks;
    }
  }

  int64_t net_change_inactive_split_blocks = 0;
  int64_t net_change_inactive_split_size = 0;

  const std::array<Block*, 2> merge_candidates = {block->m_prev, block->m_next};
  for (Block* merge_candidate : merge_candidates) {
    const int64_t subsumed_size =
        try_merge_blocks(block, merge_candidate, pool);
    if (subsumed_size > 0) {
      net_change_inactive_split_blocks -= 1;
      net_change_inactive_split_size -= subsumed_size;
    }
  }

  pool->insert(block);

  if (block->is_split()) {
    net_change_inactive_split_blocks += 1;
    net_change_inactive_split_size += block->m_size;
  }

  DeviceStats& stats = get_stats_for_device(block->m_device);
  StatTypes stat_types;
  stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
  stat_types[static_cast<size_t>(get_stat_type_for_pool(block->m_pool_type))] =
      true;
  update_stat_array(
      stats.inactive_split, net_change_inactive_split_blocks, stat_types);
  update_stat_array(
      stats.inactive_split_bytes, net_change_inactive_split_size, stat_types);
  update_stat_array(stats.active, -1, stat_types);
  update_stat_array(stats.active_bytes, -original_block_size, stat_types);
}

std::vector<const CachingDeviceAllocator::Block*> CachingDeviceAllocator::
    get_all_blocks() const {
  std::vector<const Block*> blocks;
  blocks.insert(blocks.end(), small_blocks.begin(), small_blocks.end());
  blocks.insert(blocks.end(), large_blocks.begin(), large_blocks.end());
  for (const auto& item : allocated_blocks) {
    blocks.push_back(item.second);
  }
  return blocks;
}

void CachingDeviceAllocator::insert_events(Block* block) {
  std::unordered_set<sycl::queue> queues(std::move(block->m_queue_uses));
  AT_ASSERT(block->m_queue_uses.empty());
  for (auto it = queues.begin(); it != queues.end(); ++it) {
    // removing const from reference to allow submit a barrier
    // barrier submission doesn't change the queue hash value
    sycl::queue& q = const_cast<sycl::queue&>(*it);

    auto event = queue_barrier(q);
    block->m_event_cnt++;
    dpcpp_events.emplace_back(event, block);
  }
}

void CachingDeviceAllocator::process_events() {
  while (!dpcpp_events.empty()) {
    auto& e = dpcpp_events.front();
    auto event = e.first;
    Block* block = e.second;
    bool event_completed = event.get_info<dpcpp_event_exec_stat>() ==
        dpcpp_event_cmd_stat_complete;
    if (!event_completed) {
      break;
    }
    block->m_event_cnt--;
    if (block->m_event_cnt == 0) {
      free_block(block);
    }
    dpcpp_events.pop_front();
  }
}

DeviceStats& CachingDeviceAllocator::get_stats_for_device(DeviceId device) {
  TORCH_CHECK(device >= 0);
  if ((size_t)device >= device_stats.size()) {
    device_stats.resize(device + 1);
  }
  return device_stats.at(device);
}

void CachingDeviceAllocator::cache_info_aux(
    BlockPool& blocks,
    DeviceId di,
    size_t* total,
    size_t* largest) {
  for (auto it = blocks.begin();
       it != blocks.end() && *it && (*it)->m_device == di;
       ++it) {
    size_t blocksize = (*it)->m_size;
    *total += blocksize;
    if (blocksize > *largest) {
      *largest = blocksize;
    }
  }
}

size_t CachingDeviceAllocator::try_merge_blocks(
    Block* dst,
    Block* src,
    BlockPool* pool) {
  if (!src || src->m_allocated || src->m_event_cnt > 0) {
    return 0;
  }

  AT_ASSERT(dst->is_split() && src->is_split());

  if (dst->m_prev == src) {
    dst->m_buffer = src->m_buffer;
    dst->m_prev = src->m_prev;
    if (dst->m_prev) {
      dst->m_prev->m_next = dst;
    }
  } else {
    dst->m_next = src->m_next;
    if (dst->m_next) {
      dst->m_next->m_prev = dst;
    }
  }

  const size_t subsumed_size = src->m_size;
  dst->m_size += subsumed_size;
  pool->erase(src);
  delete src;

  return subsumed_size;
}

/**
 * Note [Safe to Free Blocks on BlockPool]
 *
 * Callers must ensure that all accesses to the block, whose raw pointer is
 * allocated by SYCL APIs, have been completed before invoking sycl::free.
 *
 * We have to do a device-level synchronization before free these blocks to
 * guarantee that all kernels can access to the blocks have finished.
 */
void CachingDeviceAllocator::free_blocks(
    BlockPool& blocks,
    BlockPool::iterator it,
    BlockPool::iterator end) {
  while (it != end) {
    Block* block = *it;
    if (!block->m_prev && !block->m_next) {
      sycl::free((void*)block->m_buffer, at::xpu::get_device_context());

      DeviceStats& stats = get_stats_for_device(block->m_device);
      StatTypes stat_types;
      stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
      stat_types[static_cast<size_t>(
          get_stat_type_for_pool(block->m_pool_type))] = true;
      update_stat_array(stats.segment, -1, stat_types);
      update_stat_array(stats.reserved_bytes, -block->m_size, stat_types);

      auto cur = it;
      ++it;
      blocks.erase(cur);
      delete block;
    } else {
      ++it;
    }
  }
}

// find the boundary of the pool allocated on the specific device
void CachingDeviceAllocator::find_cached_blocks_bound(
    DeviceId di,
    BlockPool& pool,
    BlockPool::iterator& begin,
    BlockPool::iterator& end) {
  bool find_begin = false;
  bool find_end = false;
  // pool is a set stored as an ascending by device index. We would like to find
  // the blocks within the range [di, di+1).
  for (auto it = pool.begin(); it != pool.end(); it++) {
    if ((*it)->m_device == di && !find_begin) {
      // find the begin, that is the leftmost block allocated on device di.
      begin = it;
      find_begin = true;
    } else if ((*it)->m_device > di && find_begin) {
      // find the end, that is the leftmost block allocated on device larger
      // than di. It may be di+1, di+2, ... Why does it probably equals to di+2,
      // because it is possible that no allocation on di+1.
      end = it;
      find_end = true;
      break;
    }
  }
  if (!find_begin)
    begin = pool.end();
  if (!find_begin || !find_end)
    end = pool.end();
}

void CachingDeviceAllocator::free_cached_blocks(DeviceId di) {
  synchronize_and_free_events(di);

  /*
   * See Note [Safe to Free Blocks on BlockPool]
   */
  at::xpu::syncStreamsOnDevice(di);

  BlockPool::iterator begin;
  BlockPool::iterator end;
  find_cached_blocks_bound(di, large_blocks, begin, end);
  free_blocks(large_blocks, begin, end);
  find_cached_blocks_bound(di, small_blocks, begin, end);
  free_blocks(small_blocks, begin, end);

  // Release graph private pools
  for (auto it = graph_pools_freeable.begin();
       it != graph_pools_freeable.end();) {
    TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
    find_cached_blocks_bound(di, it->second->small_blocks, begin, end);
    free_blocks(it->second->small_blocks, begin, end);
    find_cached_blocks_bound(di, it->second->large_blocks, begin, end);
    free_blocks(it->second->large_blocks, begin, end);
    auto erase_count = graph_pools.erase(it->first);
    TORCH_INTERNAL_ASSERT(erase_count == 1);
    it = graph_pools_freeable.erase(it);
  }
}

void CachingDeviceAllocator::synchronize_and_free_events(
    std::optional<DeviceId> di) {
  auto remaining_events = decltype(dpcpp_events)();

  for (auto& e : dpcpp_events) {
    sycl::event event = e.first;
    Block* block = e.second;
    if (di.has_value() && block->m_device != *di) {
      remaining_events.push_back(e);
      continue;
    }
    event.wait();
    block->m_event_cnt--;
    if (block->m_event_cnt == 0) {
      free_block(block);
    }
  }

  std::swap(dpcpp_events, remaining_events);
}

void CachingDeviceAllocator::cacheInfo(
    DeviceId di,
    size_t* total,
    size_t* largest) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  cache_info_aux(large_blocks, di, total, largest);
  cache_info_aux(small_blocks, di, total, largest);
}

DeviceStats CachingDeviceAllocator::getStatsForDevice(DeviceId di) {
  assertValidDevice(di);
  std::lock_guard<std::recursive_mutex> lock(mutex);
  return get_stats_for_device(di);
}

void CachingDeviceAllocator::resetAccumulatedStats(DeviceId di) {
  assertValidDevice(di);
  std::lock_guard<std::recursive_mutex> lock(mutex);
  DeviceStats& stats = get_stats_for_device(di);

  auto reset_accumulated_stat = [](Stat& stat) {
    stat.allocated = 0;
    stat.freed = 0;
  };
  for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES);
       ++statType) {
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

void CachingDeviceAllocator::resetPeakStats(DeviceId di) {
  assertValidDevice(di);
  std::lock_guard<std::recursive_mutex> lock(mutex);
  DeviceStats& stats = get_stats_for_device(di);

  auto reset_peak_stat = [](Stat& stat) { stat.peak = stat.current; };
  for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES);
       ++statType) {
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

std::vector<SegmentInfo> CachingDeviceAllocator::snapshot() const {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  std::vector<SegmentInfo> result;
  const auto all_blocks = get_all_blocks();

  for (const Block* const head_block : all_blocks) {
    if (head_block->m_prev != nullptr) {
      continue;
    }
    result.emplace_back();
    SegmentInfo& segment_info = result.back();
    segment_info.device = head_block->m_device;
    segment_info.address = reinterpret_cast<int64_t>(head_block->m_buffer);
    segment_info.is_large = (head_block->m_pool_type == PoolType::LARGE_POOL);

    const Block* block = head_block;
    while (block != nullptr) {
      segment_info.blocks.emplace_back();
      BlockInfo& block_info = segment_info.blocks.back();

      block_info.size = block->m_size;
      block_info.allocated = block->m_allocated;
      block_info.active = block->m_allocated || (block->m_event_cnt > 0);

      segment_info.total_size += block_info.size;
      if (block_info.allocated) {
        segment_info.allocated_size += block_info.size;
      }
      if (block_info.active) {
        segment_info.active_size += block_info.size;
      }

      block = block->m_next;
    }
  }

  std::sort(
      result.begin(),
      result.end(),
      [](const SegmentInfo& a, const SegmentInfo& b) {
        if (a.device != b.device) {
          return a.device < b.device;
        }
        return a.address < b.address;
      });

  return result;
}

void CachingDeviceAllocator::dumpMemoryStatus(DeviceId deviceIndex) {
  DeviceStats& stats = get_stats_for_device(deviceIndex);
  auto& dpcppDev = at::xpu::get_raw_device(deviceIndex);
  size_t device_total = dpcppGlobalMemSize(deviceIndex);
  TORCH_WARN("GPU", deviceIndex, " memory status:");
  TORCH_WARN("Total capacity: ", format_size(device_total));
  TORCH_WARN(
      "Allocated: ",
      format_size(
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current));
  TORCH_WARN(
      "Reserved: ",
      format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                      .current));
}

// Called by XPUGraph::begin_recording
void CachingDeviceAllocator::beginAllocateToPool(
    DeviceId deviceIndex,
    MempoolId_t mempoolId,
    std::function<bool(sycl::queue*)> filter) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto search_key = std::make_pair(deviceIndex, mempoolId);
  auto it = graph_pools.find(search_key);
  if (it == graph_pools.end()) {
    graph_pools.emplace(search_key, std::make_unique<PrivatePool>());
  } else {
    TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
    it->second->use_count += 1;
  }
  for (auto it2 = recordings_underway.begin(); it2 != recordings_underway.end();
       ++it2) {
    TORCH_CHECK(
        it2->first != search_key,
        "beginAllocateToPool: already recording to mempool_id");
  }
  recordings_underway.emplace_back(search_key, std::move(filter));
}

// Called by XPUGraph::end_recording
void CachingDeviceAllocator::endAllocateToPool(
    DeviceId deviceIndex,
    MempoolId_t mempoolId) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto search_key = std::make_pair(deviceIndex, mempoolId);
  for (auto it = recordings_underway.begin(); it != recordings_underway.end();
       ++it) {
    if (it->first == search_key) {
      recordings_underway.erase(it);
      return;
    }
  }
  TORCH_CHECK(
      false, "endAllocateToPool: not currently recording to mempool_id");
}

// Called by XPUGraph::reset
void CachingDeviceAllocator::releasePool(
    DeviceId deviceIndex,
    MempoolId_t mempoolId) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto search_key = std::make_pair(deviceIndex, mempoolId);
  auto it = graph_pools.find(search_key);
  TORCH_INTERNAL_ASSERT(it != graph_pools.end());
  auto uc = --(it->second->use_count);
  TORCH_INTERNAL_ASSERT(uc >= 0);
  if (uc == 0) {
    bool inserted =
        graph_pools_freeable.insert({search_key, it->second.get()}).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
