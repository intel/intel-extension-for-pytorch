#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/xpu/XPUStream.h>
#include <runtime/CachingDeviceAllocator.h>
#include <sycl/sycl.hpp>
#include <mutex>

#include <core/AllocationInfo.h>
#include <utils/Macros.h>

namespace torch_ipex::xpu {
namespace dpcpp {
class DeviceAllocator final : public at::Allocator {
 public:
  DeviceAllocator() {}
  static DeviceAllocator* Instance();

  static void deleter(void* ptr);

  at::DataPtr allocate(size_t size) const override;
  at::DataPtr allocate(const sycl::queue& queue, size_t size) const;

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

  void recordStream(const at::DataPtr& ptr, at::xpu::XPUStream stream);

  std::mutex* getFreeMutex();

  DeviceStats getDeviceStats(at::DeviceIndex device_index);

  void resetAccumulatedStats(at::DeviceIndex device_index);

  void resetPeakStats(at::DeviceIndex device_index);

  void dumpMemoryStatus(at::DeviceIndex device_index);

  std::vector<SegmentInfo> snapshot();

 private:
  CachingDeviceAllocator* alloc();
};

void recordStreamInDevAlloc(const at::DataPtr& ptr, at::xpu::XPUStream stream);

} // namespace dpcpp
} // namespace torch_ipex::xpu
