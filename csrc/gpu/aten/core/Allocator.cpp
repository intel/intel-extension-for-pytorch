#include <core/Allocator.h>
#include <core/Stream.h>
#include <runtime/CachingDeviceAllocator.h>
#include <runtime/CachingHostAllocator.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include "DeviceAllocator.h"
#include "HostAllocator.h"

namespace xpu {
namespace dpcpp {

void DeviceAllocator::deleter(void* ptr) {
  auto* ctx = static_cast<at::AtenIpexTypeXPU::DPCPPTensorContext*>(ptr);
  auto data = ctx->data();
  Instance()->alloc()->free(data);
  delete ctx;
}

DataPtr DeviceAllocator::allocate(size_t size) const {
  DeviceIndex curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  void* r = nullptr;
  if (size != 0) {
    auto stream = getCurrentDPCPPStream(curDevID);
    Instance()->alloc()->malloc(&r, size, &dpcppGetQueueFromStream(stream));
  }
  auto ctx = new at::AtenIpexTypeXPU::DPCPPTensorContext(r);
  return {r, ctx, &deleter, Device(DeviceType::XPU, curDevID)};
}

void* DeviceAllocator::raw_allocate(size_t size) {
  DeviceIndex curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  void* r = nullptr;
  if (size != 0) {
    auto stream = getCurrentDPCPPStream(curDevID);
    Instance()->alloc()->malloc(&r, size, &dpcppGetQueueFromStream(stream));
  }
  return r;
}

at::DeleterFnPtr DeviceAllocator::raw_deleter() const {
  return &deleter;
}

void DeviceAllocator::emptyCache() {
  alloc()->emptyCache();
}

void DeviceAllocator::cacheInfo(
    DeviceId deviceIndex,
    size_t* cachedAndFree,
    size_t* largestBlock) {
  alloc()->cacheInfo(deviceIndex, cachedAndFree, largestBlock);
}

void* DeviceAllocator::getBaseAllocation(void* ptr, size_t* size) {
  return alloc()->getBaseAllocation(ptr, size);
}

void DeviceAllocator::recordStream(const at::DataPtr& ptr, DPCPPStream stream) {
  if (!ptr.get()) {
    return;
  }

  if (ptr.get_deleter() != &deleter) {
    return;
  }

  alloc()->recordQueue(ptr.get(), &dpcppGetQueueFromStream(stream));
}

std::mutex* DeviceAllocator::getFreeMutex() {
  return alloc()->getDPCPPFreeMutex();
}

DeviceStats DeviceAllocator::getDeviceStats(DeviceIndex device_index) {
  return alloc()->getStatsForDevice(device_index);
}

void DeviceAllocator::resetAccumulatedStats(DeviceIndex device_index) {
  alloc()->resetAccumulatedStats(device_index);
}

void DeviceAllocator::resetPeakStats(DeviceIndex device_index) {
  alloc()->resetPeakStats(device_index);
}

void DeviceAllocator::dumpMemoryStatus(DeviceIndex device_index) {
  alloc()->dumpMemoryStatus(device_index);
}

std::vector<SegmentInfo> DeviceAllocator::snapshot() {
  return alloc()->snapshot();
}

CachingDeviceAllocator* DeviceAllocator::alloc() {
  return CachingDeviceAllocator::Instance();
}

static DeviceAllocator myInstance;

REGISTER_ALLOCATOR(kXPU, &myInstance);

DeviceAllocator* DeviceAllocator::Instance() {
  return &myInstance;
}

at::Allocator* getDeviceAllocator() {
  return DeviceAllocator::Instance();
}

void emptyCacheInDevAlloc() {
  DeviceAllocator::Instance()->emptyCache();
}

void cacheInfoFromDevAlloc(
    DeviceIndex deviceIndex,
    size_t* cachedAndFree,
    size_t* largestBlock) {
  DeviceAllocator::Instance()->cacheInfo(
      deviceIndex, cachedAndFree, largestBlock);
}

void* getBaseAllocationFromDevAlloc(void* ptr, size_t* size) {
  return DeviceAllocator::Instance()->getBaseAllocation(ptr, size);
}

void recordStreamInDevAlloc(const DataPtr& ptr, DPCPPStream stream) {
  DeviceAllocator::Instance()->recordStream(ptr, stream);
}

DeviceStats getDeviceStatsFromDevAlloc(DeviceIndex device_index) {
  return DeviceAllocator::Instance()->getDeviceStats(device_index);
}

void resetAccumulatedStatsInDevAlloc(DeviceIndex device_index) {
  DeviceAllocator::Instance()->resetAccumulatedStats(device_index);
}

void resetPeakStatsInDevAlloc(DeviceIndex device_index) {
  DeviceAllocator::Instance()->resetPeakStats(device_index);
}

void dumpMemoryStatusFromDevAlloc(DeviceIndex device_index) {
  DeviceAllocator::Instance()->dumpMemoryStatus(device_index);
}

std::vector<SegmentInfo> snapshotOfDevAlloc() {
  return DeviceAllocator::Instance()->snapshot();
}

std::mutex* getFreeMutexOfDevAlloc() {
  return DeviceAllocator::Instance()->getFreeMutex();
}

/// Host Allocator
HostAllocator* HostAllocator::Instance() {
  static HostAllocator myInstance;
  return &myInstance;
}

void HostAllocator::deleter(void* ptr) {
  Instance()->release(ptr);
}

at::DataPtr HostAllocator::allocate(size_t size) const {
  void* ptr = nullptr;
  Instance()->alloc()->malloc(&ptr, size);
  return {ptr, ptr, &deleter, at::DeviceType::CPU};
}

void* HostAllocator::raw_allocate(size_t size) {
  void* ptr = nullptr;
  Instance()->alloc()->malloc(&ptr, size);
  return ptr;
}

at::DeleterFnPtr HostAllocator::raw_deleter() const {
  return &deleter;
}

bool HostAllocator::isHostPtr(const void* ptr) {
  return alloc()->isHostPtr(ptr);
}

void HostAllocator::emptyCache() {
  alloc()->emptyCache();
}

void HostAllocator::recordEvent(void* ptr, sycl::event& e) {
  alloc()->recordEvent(ptr, e);
}

CachingHostAllocator* HostAllocator::alloc() {
  return CachingHostAllocator::Instance();
}

void HostAllocator::release(void* ptr) {
  alloc()->release(ptr);
}

Allocator* getHostAllocator() {
  return HostAllocator::Instance();
}

void emptyCacheInHostAlloc() {
  HostAllocator::Instance()->emptyCache();
}

bool isAllocatedByHostAlloc(const void* ptr) {
  return HostAllocator::Instance()->isHostPtr(ptr);
}

} // namespace dpcpp
} // namespace xpu
