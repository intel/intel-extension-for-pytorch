#include <core/Event.h>
#include <core/GuardImpl.h>
#include <include/xpu/Stream.h>
#include "Allocator.h"

namespace xpu {
namespace dpcpp {
namespace impl {

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
  return getCurrentDPCPPStream(d.index()).unwrap();
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

void DPCPPGuardImpl::destroyEvent(void* event, const DeviceIndex device_index)
    const noexcept {
  if (!event)
    return;

  DPCPPEvent* dpcpp_event = static_cast<DPCPPEvent*>(event);
  delete dpcpp_event;
}

void DPCPPGuardImpl::record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const c10::EventFlag flag) const {
  TORCH_CHECK(
      device_index == -1 || device_index == stream.device_index(),
      "Event device index ",
      device_index,
      " does not match recording stream's device index ",
      stream.device_index(),
      ".");

  if (*event == nullptr) {
    // lazy init
    *event = new DPCPPEvent();
  }

  DPCPPEvent* dpcpp_event = static_cast<DPCPPEvent*>(*event);
  DPCPPStream dpcpp_stream{stream};

  dpcpp_event->record(dpcpp_stream);
}

void DPCPPGuardImpl::block(void* event, const Stream& stream) const {
  if (!event)
    return;

  DPCPPEvent* dpcpp_event = static_cast<DPCPPEvent*>(event);
  DPCPPStream dpcpp_stream{stream};

  dpcpp_event->block(dpcpp_stream);
}

bool DPCPPGuardImpl::queryEvent(void* event) const {
  if (!event)
    return true;

  DPCPPEvent* dpcpp_event = static_cast<DPCPPEvent*>(event);
  return dpcpp_event->query();
}

// Stream-related functions
bool DPCPPGuardImpl::queryStream(const Stream& stream) const {
  TORCH_CHECK(false, "xpu not support queryStream so far");
  // TODO: add work around to enable the queue query by tracking the last kernel
  // event.
  //  DPCPPStream dpcpp_stream{stream};
  //  auto& queue = get_queue_from_stream(stream);
  //  queue.get_info<sycl::info::queue::>();
  //  return dpcpp_stream.query();
  return false;
}

Stream DPCPPGuardImpl::getStreamFromGlobalPool(
    Device device,
    bool is_high_priority) const {
  TORCH_CHECK(
      is_high_priority == false, "xpu doesn't support prioritized steam");

  DPCPPStream dpcpp_stream =
      xpu::dpcpp::getDPCPPStreamFromPool(false, device.index());
  return dpcpp_stream.unwrap();
}

void DPCPPGuardImpl::synchronizeStream(const Stream& stream) const {
  auto& queue = get_queue_from_stream(stream);
  queue.wait_and_throw();
  return;
}

void DPCPPGuardImpl::recordDataPtrOnStream(
    const c10::DataPtr& data_ptr,
    const Stream& stream) const {
  recordStreamInDevAlloc(data_ptr, DPCPPStream(stream));
}

} // namespace impl
} // namespace dpcpp
} // namespace xpu
