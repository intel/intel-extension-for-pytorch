#pragma once

#include <c10/util/Optional.h>
#include <utils/DPCPP.h>
#include <core/Stream.h>

namespace xpu {
namespace dpcpp {

/*
* DPCPPEvents are movable not copyable wrappers around SYCL's events.
*
* DPCPPEvents are constructed lazily when first recorded unless it is
* reconstructed from a IpcEventHandle_t. The event has a device, and this
* device is acquired from the first recording stream. However, if reconstructed
* from a handle, the device should be explicitly specified; or if ipc_handle() is
* called before the event is ever recorded, it will use the current device.
* Later streams that record the event must match this device.
*/
struct  DPCPPEvent {
  // Constructors
  DPCPPEvent() {}

  ~DPCPPEvent() {}

  DPCPPEvent(const DPCPPEvent&) = delete;
  DPCPPEvent& operator=(const DPCPPEvent&) = delete;

  DPCPPEvent(DPCPPEvent&& other);

  DPCPPEvent& operator=(DPCPPEvent&& other);

  at::optional<at::Device> device() const;

  bool isCreated() const;

  DeviceIndex device_index() const;

  std::vector<DPCPP::event> event() const;

  bool query() const;

  void record();

  void record(const DPCPPStream& stream);

  void recordOnce(const DPCPPStream& stream);

  void block(const DPCPPStream& stream);

  void synchronize();

  float elapsed_time(const DPCPPEvent& other) const;

  void ipc_handle(void * handle);

private:
  DeviceIndex device_index_ = -1;
  std::vector<DPCPP::event> events_;

  void moveHelper(DPCPPEvent&& other);
};

} // namespace dpcpp
} // namespace xpu
