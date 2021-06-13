#pragma once

#include <runtime/DPCPP.h>
#include <core/Stream.h>

#include <cstdint>
#include <utility>

namespace xpu { namespace dpcpp {

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

#if 0
  DPCPPEvent(
      DeviceIndex device_index, const xpuIpcEventHandle_t* handle) {
      device_index_ = device_index;
      DPCPPGuard guard(device_index_);

      AT_CHECK(xpuIpcOpenEventHandle(&event_, *handle));
      is_created_ = true;
  }
#endif

  ~DPCPPEvent() {}

  DPCPPEvent(const DPCPPEvent&) = delete;
  DPCPPEvent& operator=(const DPCPPEvent&) = delete;

  DPCPPEvent(DPCPPEvent&& other) { moveHelper(std::move(other)); }
  DPCPPEvent& operator=(DPCPPEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

//  operator xpuEvent_t() const { return event(); }

  // Less than operator (to allow use in sets)
//  friend bool operator<(const DPCPPEvent& left, const DPCPPEvent& right) {
//    return left.event_ < right.event_;
//  }

  optional<at::Device> device() const {
    if (!events_.empty()) {
      return at::Device(at::kXPU, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return !events_.empty(); }
  DeviceIndex device_index() const {return device_index_;}
  std::vector<DPCPP::event> event() const { return events_; }

  bool query() const {
    if (events_.empty()) {
      return true;
    }

    for (auto& event : events_) {
      auto ret = event.get_info<DPCPP::info::event::command_execution_status>();
      if (ret != DPCPP::info::event_command_status::complete) {
        return false;
      }
    }

    return true;
  }

  void record() { record(getCurrentDPCPPStream()); }

  void recordOnce(const DPCPPStream& stream) {
    if (events_.empty()) record(stream);
  }

  void record(const DPCPPStream& stream) {
    if (events_.empty()) {
      device_index_ = stream.device_index();
    } else {
      TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
                  " does not match recording stream's device ", stream.device_index(), ".");
    }

    events_.push_back(stream.dpcpp_queue().submit_barrier());
  }

  void block(const DPCPPStream& stream) {
    if (!events_.empty()) {
       stream.dpcpp_queue().submit_barrier(events_);
    }
  }

  void synchronize() {
    if (!events_.empty()) {
      for (auto& event : events_) {
        event.wait_and_throw();
      }
    }
  }

  float elapsed_time(const DPCPPEvent& other) const {
    TORCH_CHECK(isCreated() && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(query() && other.query(),
      "Both events must be completed before calculating elapsed time.");

    float time_ms = 0;
    auto self_last = *events_.rbegin();
    auto other_last = *other.events_.rbegin();
    auto self_end = self_last.template get_profiling_info<
            cl::sycl::info::event_profiling::command_end>();
    auto other_end = other_last.template get_profiling_info<
            cl::sycl::info::event_profiling::command_end>();
    if (other_end <= self_end) {
      // nanoseconds to milliseconds
      time_ms = (self_end - other_end) / (1000.0 * 1000.0);
    } else {
      // nanoseconds to milliseconds
      time_ms = -((other_end - self_end) / (1000.0 * 1000.0));
    }
    return time_ms;
  }

  void ipc_handle(void * handle) {
      AT_ERROR("ipc_handle with DPCPP is not supported");
  }

private:
  DeviceIndex device_index_ = -1;
  std::vector<DPCPP::event> events_;

  void moveHelper(DPCPPEvent&& other) {
    std::swap(device_index_, other.device_index_);
    std::swap(events_, other.events_);
  }
};

} // namespace dpcpp
} // namespace xpu
