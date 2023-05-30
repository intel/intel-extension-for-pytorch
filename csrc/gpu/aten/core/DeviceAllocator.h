#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <runtime/CachingDeviceAllocator.h>
#include <mutex>

#include <core/AllocationInfo.h>
#include <core/Stream.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {
class DeviceAllocator final : public at::Allocator {
 public:
  DeviceAllocator() {}
  static DeviceAllocator* Instance();

  static void deleter(void* ptr);

  DataPtr allocate(size_t size) const override;

  /*
    `raw_allocate` is an IPEX private malloc utils. It is expected to be
    used only in special cases. API `allocate` that Torch defines always
    has higher priority. Please re-assure you really need `raw_allocate`
    before using it.
  */
  void* raw_allocate(size_t size);

  at::DeleterFnPtr raw_deleter() const override;

  void emptyCache();

  void cacheInfo(
      DeviceId deviceIndex,
      size_t* cachedAndFree,
      size_t* largestBlock);

  void* getBaseAllocation(void* ptr, size_t* size);

  void recordStream(const at::DataPtr& ptr, DPCPPStream stream);

  std::mutex* getFreeMutex();

  DeviceStats getDeviceStats(DeviceIndex device_index);

  void resetAccumulatedStats(DeviceIndex device_index);

  void resetPeakStats(DeviceIndex device_index);

  void dumpMemoryStatus(DeviceIndex device_index);

  std::vector<SegmentInfo> snapshot();

 private:
  CachingDeviceAllocator* alloc();
};
} // namespace dpcpp
} // namespace xpu
