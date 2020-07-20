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
  NUM_TYPES = 3  // remember to update this whenever a new stat type is added
};

typedef std::array<CAStat, static_cast<size_t>(CAStatType::NUM_TYPES)> CAStatArray;

// Struct containing memory allocator summary statistics for a device.
struct CADeviceStats {
  // COUNT: allocations requested by client code
  CAStatArray allocation;
  // COUNT: number of allocated segments from dpcpp_malloc().
  CAStatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  CAStatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be released via dpcpp_free)
  CAStatArray inactive_split;

  // SUM: bytes requested by client code
  CAStatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  CAStatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  CAStatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  CAStatArray inactive_split_bytes;

  // COUNT: total number of failed calls to CUDA malloc necessitating cache flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
  int64_t num_ooms = 0;
};

// Struct containing info of an allocation block (i.e. a fractional part of a cudaMalloc)..
struct CABlockInfo {
  int64_t size = 0;
  bool allocated = false;
  bool active = false;
};

// Struct containing info of a memory segment (i.e. one contiguous dpcppMalloc).
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
std::vector<CASegmentInfo> dpcpp_snapshot();
std::mutex* getFreeMutex();

}} // namespace at::dpcpp

#endif
