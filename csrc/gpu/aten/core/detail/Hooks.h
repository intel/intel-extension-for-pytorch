#pragma once

#include <ATen/Generator.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/dlpack.h>

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
      const DLDevice_& dl_device,
      void* data) const override;
  DLDevice_& getDLPackDeviceFromATenDevice(
      DLDevice_& dl_device,
      const at::Device& aten_device,
      void* data) const override;
};

} // namespace detail
} // namespace dpcpp
} // namespace xpu
