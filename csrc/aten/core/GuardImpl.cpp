#include <core/GuardImpl.h>

namespace xpu {
namespace dpcpp {
namespace impl {

constexpr DeviceType DPCPPGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(XPU, DPCPPGuardImpl);


DPCPPGuardImpl::DPCPPGuardImpl() {}

DPCPPGuardImpl::DPCPPGuardImpl(DeviceType t) {
  TORCH_INTERNAL_ASSERT(t == DeviceType::XPU);
}

DeviceType DPCPPGuardImpl::type() const {
  return DeviceType::XPU;
}

Device DPCPPGuardImpl::exchangeDevice(Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == DeviceType::XPU);
  Device old_device = getDevice();
  if (old_device.index() != d.index()) {
    set_device(d.index());
  }
  return old_device;
}

Device DPCPPGuardImpl::getDevice() const {
  return Device(DeviceType::XPU, current_device());
}

void DPCPPGuardImpl::setDevice(Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == DeviceType::XPU);
  set_device(d.index());
}

void DPCPPGuardImpl::uncheckedSetDevice(Device d) const noexcept {
  set_device(d.index());
}

Stream DPCPPGuardImpl::getStream(Device d) const noexcept {
  return getCurrentDPCPPStream().unwrap();
}

Stream DPCPPGuardImpl::exchangeStream(Stream s) const noexcept {
  DPCPPStream cs(s);
  auto old_stream = getCurrentDPCPPStream(s.device().index());
  setCurrentDPCPPStream(cs);
  return old_stream.unwrap();
}

DeviceIndex DPCPPGuardImpl::deviceCount() const noexcept {
  return device_count();
}

} // namespace impl
} // namespace dpcpp
} // namespace xpu
