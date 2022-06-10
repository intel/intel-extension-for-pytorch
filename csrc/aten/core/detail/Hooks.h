#pragma once

#include <ATen/Generator.h>
#include <ATen/detail/XPUHooksInterface.h>

namespace xpu {
namespace dpcpp {
namespace detail {

// The real implementation of XPUHooksInterface
struct XPUHooks : public at::XPUHooksInterface {
  XPUHooks(at::XPUHooksArgs) {}
  void initXPU() const override;
  bool hasXPU() const override;
  std::string showConfig() const override;
  at::Device getATenDeviceFromDLPackDevice(
      const DLDevice& dl_device,
      void* data) const override;
  DLDevice getDLPackDeviceFromATenDevice(
      const at::Device& aten_device,
      void* data) const override;
};

} // namespace detail
} // namespace dpcpp
} // namespace xpu
