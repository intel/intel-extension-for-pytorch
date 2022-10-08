#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <core/AllocationInfo.h>
#include <core/Stream.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

/// Device Allocator
void emptyCacheInDevAlloc();

DeviceStats getDeviceStatsFromDevAlloc(DeviceIndex device_index);

void resetAccumulatedStatsInDevAlloc(DeviceIndex device_index);

void resetPeakStatsInDevAlloc(DeviceIndex device_index);

std::vector<SegmentInfo> snapshotOfDevAlloc();

at::Allocator* getDeviceAllocator();

void cacheInfoFromDevAlloc(
    DeviceIndex deviceIndex,
    size_t* cachedAndFree,
    size_t* largestBlock);

void* getBaseAllocationFromDevAlloc(void* ptr, size_t* size);

void recordStreamInDevAlloc(const DataPtr& ptr, DPCPPStream stream);

void dumpMemoryStatusFromDevAlloc(DeviceIndex device_index);

std::mutex* getFreeMutexOfDevAlloc();

/// Host Allocator
// Provide a caching allocator for host allocation by USM malloc_host
Allocator* getHostAllocator();

// Record the event on queue where the host allocation is using
void recordEventInHostAlloc(void* ptr, sycl::event& e);

// Releases all cached host memory allocations
void emptyCacheInHostAlloc();

bool isAllocatedByHostAlloc(void* ptr);

} // namespace dpcpp
} // namespace xpu
