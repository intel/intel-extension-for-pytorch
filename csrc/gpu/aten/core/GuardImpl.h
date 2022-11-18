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

  Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const override;

  Stream exchangeStream(Stream s) const noexcept override;

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override;

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const c10::EventFlag flag) const override;

  void block(void* event, const Stream& stream) const override;

  bool queryEvent(void* event) const override;

  DeviceIndex deviceCount() const noexcept override;

  bool queryStream(const Stream& stream) const override;

  void synchronizeStream(const Stream& stream) const override;

  void recordDataPtrOnStream(const c10::DataPtr&, const Stream&) const override;
};
} // namespace impl
} // namespace dpcpp
} // namespace xpu
