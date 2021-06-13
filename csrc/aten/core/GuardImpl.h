#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <runtime/DPCPPUtils.h>
#include <runtime/Exception.h>
#include <core/Device.h>
#include <core/Stream.h>


using namespace xpu::dpcpp;

namespace xpu {
namespace dpcpp {
namespace impl {

struct DPCPPGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::XPU;
  DPCPPGuardImpl() {}
  DPCPPGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::XPU);
  }
  DeviceType type() const override {
    return DeviceType::XPU;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::XPU);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      AT_DPCPP_CHECK(dpcppSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    DeviceIndex device;
    AT_DPCPP_CHECK(dpcppGetDevice(&device));
    return Device(DeviceType::XPU, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::XPU);
    AT_DPCPP_CHECK(dpcppSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    int __err = dpcppSetDevice(d.index());
    if (__err != DPCPP_SUCCESS) {
      TORCH_WARN("DPCPP error: uncheckedSetDevice failed");
    }
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentDPCPPStream().unwrap();
  }
  Stream exchangeStream(Stream s) const noexcept override {
    DPCPPStream cs(s);
    auto old_stream = getCurrentDPCPPStream(s.device().index());
    setCurrentDPCPPStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }
};
} // namespace impl
} // namespace dpcpp
} // namespace at
