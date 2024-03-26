#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <mutex>

#include <core/AllocationInfo.h>
#include <utils/Macros.h>

namespace torch_ipex::xpu {
namespace dpcpp {

/// Device Allocator
IPEX_API void emptyCacheInDevAlloc();

IPEX_API DeviceStats getDeviceStatsFromDevAlloc(at::DeviceIndex device_index);

IPEX_API void resetAccumulatedStatsInDevAlloc(at::DeviceIndex device_index);

IPEX_API void resetPeakStatsInDevAlloc(at::DeviceIndex device_index);

IPEX_API std::vector<SegmentInfo> snapshotOfDevAlloc();

at::Allocator* getDeviceAllocator();

void cacheInfoFromDevAlloc(
    at::DeviceIndex deviceIndex,
    size_t* cachedAndFree,
    size_t* largestBlock);

void* getBaseAllocationFromDevAlloc(void* ptr, size_t* size);

IPEX_API void dumpMemoryStatusFromDevAlloc(at::DeviceIndex device_index);

std::mutex* getFreeMutexOfDevAlloc();

/// Host Allocator
// Provide a caching allocator for host allocation by USM malloc_host
at::Allocator* getHostAllocator();

// Releases all cached host memory allocations
void emptyCacheInHostAlloc();

bool isAllocatedByHostAlloc(const void* ptr);

} // namespace dpcpp
} // namespace torch_ipex::xpu
