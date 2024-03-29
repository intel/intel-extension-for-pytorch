#include <core/Allocator.h>
#include <runtime/CachingDeviceAllocator.h>
#include <runtime/CachingHostAllocator.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include "DeviceAllocator.h"
#include "HostAllocator.h"

namespace torch_ipex::xpu {
namespace dpcpp {

void DeviceAllocator::deleter(void* ptr) {
  auto* ctx = static_cast<at::AtenIpexTypeXPU::DPCPPTensorContext*>(ptr);
  auto data = ctx->data();
  Instance()->alloc()->free(data);
  delete ctx;
}

DataPtr DeviceAllocator::allocate(size_t size) {
  DeviceIndex curDevID = at::xpu::current_device();
  void* r = nullptr;
  if (size != 0) {
    auto stream = at::xpu::getCurrentXPUStream(curDevID);
    Instance()->alloc()->malloc(&r, size, &stream.queue());
  }
  auto ctx = new at::AtenIpexTypeXPU::DPCPPTensorContext(r);
  return {r, ctx, &deleter, Device(DeviceType::XPU, curDevID)};
}

DataPtr DeviceAllocator::allocate(const sycl::queue& queue, size_t size) const {
  void* r = nullptr;
  DeviceIndex devID = -1;
  for (auto i = 0; i < at::xpu::device_count(); i++) {
    if (at::xpu::get_raw_device(i) == queue.get_device()) {
      devID = i;
      break;
    }
  }
  TORCH_CHECK(devID != -1, "Unrecognized queue in DeviceAllocator::allocate.");
  if (size != 0) {
    Instance()->alloc()->malloc(&r, size, &const_cast<sycl::queue&>(queue));
  }
  auto ctx = new at::AtenIpexTypeXPU::DPCPPTensorContext(r);
  return {r, ctx, &deleter, Device(DeviceType::XPU, devID)};
}

void* DeviceAllocator::raw_allocate(size_t size) {
  DeviceIndex curDevID = at::xpu::current_device();
  void* r = nullptr;
  if (size != 0) {
    auto stream = at::xpu::getCurrentXPUStream(curDevID);
    Instance()->alloc()->malloc(&r, size, &stream.queue());
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

void DeviceAllocator::recordStream(
    const at::DataPtr& ptr,
    at::xpu::XPUStream stream) {
  if (!ptr.get()) {
    return;
  }

  if (ptr.get_deleter() != &deleter) {
    return;
  }

  alloc()->recordQueue(ptr.get(), &stream.queue());
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

void DeviceAllocator::copy_data(void* dest, const void* src, std::size_t count)
    const {
  at::xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
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

void recordStreamInDevAlloc(const DataPtr& ptr, at::xpu::XPUStream stream) {
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

at::DataPtr HostAllocator::allocate(size_t size) {
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

void HostAllocator::copy_data(void* dest, const void* src, std::size_t count)
    const {
  at::xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
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
} // namespace torch_ipex::xpu
