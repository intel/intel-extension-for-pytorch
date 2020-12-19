#ifndef DPCPP_HOOK_IMPL_H
#define DPCPP_HOOK_IMPL_H

#ifdef USE_USM
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/Generator.h>

namespace at { namespace dpcpp { namespace detail {

// The real implementation of DPCPPHooksInterface
struct DPCPPHooks : public at::XPUHooksInterface {
  DPCPPHooks(at::XPUHooksArgs) {}
  void initXPU() const override;
  bool hasXPU() const override;
  bool hasOneMKL() const override;
  bool hasOneDNN() const override;
  std::string showConfig() const override;
  int64_t getCurrentDevice() const override;
  int getDeviceCount() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(void* data) const override;
  Allocator* getPinnedMemoryAllocator() const override;
  const Generator& getDefaultXPUGenerator(DeviceIndex device_index = -1) const override;
};

}}} // at::dpcpp::detail

#endif
#endif // DPCPP_HOOK_IMPL_H
