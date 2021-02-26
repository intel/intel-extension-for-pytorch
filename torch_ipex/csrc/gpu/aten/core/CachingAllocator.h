#ifndef CACHING_ALLOCATOR_H
#define CACHING_ALLOCATOR_H

#include <ATen/Context.h>
#include <ATen/core/ATenGeneral.h>

#include <core/DPCPPUtils.h>
#include <core/Functions.h>
#include <core/Memory.h>
#include <core/Stream.h>

namespace at {
namespace dpcpp {

class DPCPPOutOfMemoryError : public c10::Error {
    using Error::Error;
};

struct CAStat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct CAStatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3  
};

typedef std::array<CAStat, static_cast<size_t>(CAStatType::NUM_TYPES)> CAStatArray;

struct CADeviceStats {
  CAStatArray allocation;
  CAStatArray segment;
  CAStatArray active;
  CAStatArray inactive_split;
  CAStatArray allocated_bytes;
  CAStatArray reserved_bytes;
  CAStatArray active_bytes;
  CAStatArray inactive_split_bytes;
  
  int64_t num_alloc_retries = 0;
  int64_t num_ooms = 0;
};

struct CABlockInfo {
  int64_t size = 0;
  bool allocated = false;
  bool active = false;
};

struct CASegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  std::vector<CABlockInfo> blocks;
};

void* dpcpp_raw_alloc(size_t nbytes);
void* dpcpp_raw_alloc_with_queue(size_t nbytes, DPCPP::queue &queue);
void dpcpp_raw_delete(void* ptr);

Allocator* dpcpp_getCachingAllocator();
void dpcpp_emptyCache();
void dpcpp_cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
void* dpcpp_getBaseAllocation(void *ptr, size_t *size);
void dpcpp_recordQueue(const DataPtr&, at::dpcpp::DPCPPStream stream);
CADeviceStats dpcpp_getDeviceStats(int device);
void dpcpp_resetAccumulatedStats(int device);
void dpcpp_resetPeakStats(int device);
void dpcpp_dumpMemoryStatus(int deviceIndex);
std::vector<CASegmentInfo> dpcpp_snapshot();
std::mutex* getFreeMutex();

}} // namespace at::dpcpp

#endif
