#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <core/Device.h>
#include <core/Stream.h>

using namespace xpu::dpcpp;

namespace xpu {
namespace dpcpp {
namespace impl {

struct DPCPPGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::XPU;

  DPCPPGuardImpl();

  DPCPPGuardImpl(DeviceType t);

  DeviceType type() const override;

  Device exchangeDevice(Device d) const override;

  Device getDevice() const override;

  void setDevice(Device d) const override;

  void uncheckedSetDevice(Device d) const noexcept override;

  Stream getStream(Device d) const noexcept override;

  Stream exchangeStream(Stream s) const noexcept override;

  DeviceIndex deviceCount() const noexcept override;
};
} // namespace impl
} // namespace dpcpp
} // namespace at
