#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <core/SYCLStream.h>
#include <core/SYCLUtils.h>
#include <core/SYCLException.h>
#include <core/SYCLFunctions.h>


namespace c10 {
namespace sycl {
namespace impl {

struct SYCLGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::DPCPP;
  SYCLGuardImpl() {}
  SYCLGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::DPCPP);
  }
  DeviceType type() const override {
    return DeviceType::DPCPP;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::DPCPP);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_SYCL_CHECK(syclSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    DeviceIndex device;
    C10_SYCL_CHECK(syclGetDevice(&device));
    return Device(DeviceType::DPCPP, device);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::DPCPP);
    C10_SYCL_CHECK(syclSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    int __err = syclSetDevice(d.index());
    if (__err != SYCL_SUCCESS) {
      AT_WARN("SYCL error: uncheckedSetDevice failed");
    }
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentSYCLStream().unwrap();
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    SYCLStream cs(s);
    auto old_stream = getCurrentSYCLStream(s.device().index());
    setCurrentSYCLStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }
};

}}} // namespace c10::sycl::impl
