#include <core/Event.h>

#include <cstdint>
#include <utility>

#include <utils/Helpers.h>

namespace xpu {
namespace dpcpp {

#if 0
  DPCPPEvent(
      DeviceIndex device_index, const xpuIpcEventHandle_t* handle) {
      device_index_ = device_index;
      DPCPPGuard guard(device_index_);

      AT_CHECK(xpuIpcOpenEventHandle(&event_, *handle));
      is_created_ = true;
  }

  operator xpuEvent_t() const { return event(); }

  Less than operator (to allow use in sets)
  friend bool operator<(const DPCPPEvent& left, const DPCPPEvent& right) {
    return left.event_ < right.event_;
  }
#endif

DPCPPEvent::DPCPPEvent(sycl::event event) {
  events_.push_back(event);
}

DPCPPEvent::DPCPPEvent(DPCPPEvent&& other) {
  moveHelper(std::move(other));
}

DPCPPEvent& DPCPPEvent::operator=(DPCPPEvent&& other) {
  moveHelper(std::move(other));
  return *this;
}

at::optional<at::Device> DPCPPEvent::device() const {
  if (!events_.empty()) {
    return at::Device(at::kXPU, device_index_);
  } else {
    return {};
  }
}

bool DPCPPEvent::isCreated() const {
  return !events_.empty();
}

DeviceIndex DPCPPEvent::device_index() const {
  return device_index_;
}

std::vector<sycl::event> DPCPPEvent::event() const {
  return events_;
}

bool DPCPPEvent::query() const {
  if (events_.empty()) {
    return true;
  }

  for (auto& event : events_) {
    auto ret = event.get_info<dpcpp_event_exec_stat>();
    if (ret != dpcpp_event_cmd_stat_complete) {
      return false;
    }
  }

  return true;
}

void DPCPPEvent::record() {
  record(getCurrentDPCPPStream());
}

void DPCPPEvent::record(const DPCPPStream& stream) {
  if (events_.empty()) {
    device_index_ = stream.device_index();
  } else {
    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");
  }

  events_.push_back(xpu::dpcpp::queue_barrier(stream.dpcpp_queue()));
}

void DPCPPEvent::recordOnce(const DPCPPStream& stream) {
  if (events_.empty())
    record(stream);
}

void DPCPPEvent::block(const DPCPPStream& stream) {
  if (!events_.empty()) {
    xpu::dpcpp::queue_barrier(stream.dpcpp_queue(), events_);
  }
}

void DPCPPEvent::synchronize() {
  if (!events_.empty()) {
    for (auto& event : events_) {
      event.wait_and_throw();
    }
  }
}

float DPCPPEvent::elapsed_time(const DPCPPEvent& other) const {
  TORCH_CHECK(
      isCreated() && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
  TORCH_CHECK(
      query() && other.query(),
      "Both events must be completed before calculating elapsed time.");

  float time_ms = 0;
  auto self_last = *events_.rbegin();
  auto other_last = *other.events_.rbegin();
  long double self_end =
      self_last.template get_profiling_info<dpcpp_event_profiling_end>();
  long double other_end =
      other_last.template get_profiling_info<dpcpp_event_profiling_end>();
  // nanoseconds to milliseconds
  time_ms = (other_end - self_end) / (1000.0 * 1000.0);
  return time_ms;
}

void DPCPPEvent::ipc_handle(void* handle) {
  AT_ERROR("ipc_handle with DPCPP is not supported");
}

void DPCPPEvent::moveHelper(DPCPPEvent&& other) {
  std::swap(device_index_, other.device_index_);
  std::swap(events_, other.events_);
}

} // namespace dpcpp
} // namespace xpu
