#pragma once

#include <c10/core/Device.h>
#include <c10/core/Allocator.h>

#include <utils/Macros.h>
#include <core/Stream.h>
#include <core/AllocationInfo.h>

namespace xpu {
namespace dpcpp {

/// Device Allocator
IPEX_API void emptyCacheInDevAlloc();

IPEX_API DeviceStats getDeviceStatsFromDevAlloc(DeviceIndex device_index);

IPEX_API void resetAccumulatedStatsInDevAlloc(DeviceIndex device_index);

IPEX_API void resetPeakStatsInDevAlloc(DeviceIndex device_index);

IPEX_API std::vector<SegmentInfo> snapshotOfDevAlloc();

at::Allocator* getDeviceAllocator();

void cacheInfoFromDevAlloc(DeviceIndex deviceIndex, size_t* cachedAndFree, size_t* largestBlock);

void* getBaseAllocationFromDevAlloc(void *ptr, size_t *size);

void recordStreamInDevAlloc(const DataPtr& ptr, DPCPPStream stream);

void dumpMemoryStatusFromDevAlloc(DeviceIndex device_index);

std::mutex* getFreeMutexOfDevAlloc();

/// Host Allocator
// Provide a caching allocator for host allocation by USM malloc_host
Allocator* getHostAllocator();

// Record the event on queue where the host allocation is using
void recordEventInHostAlloc(void* ptr, DPCPP::event& e);

// Releases all cached host memory allocations
void emptyCacheInHostAlloc();

bool isAllocatedByHostAlloc(void* ptr);

} // namespace dpcpp
} // namespace xpu
