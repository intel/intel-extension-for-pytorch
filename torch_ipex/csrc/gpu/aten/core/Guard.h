#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <core/GuardImpl.h>
#include <core/Macros.h>
#include <cstddef>


namespace xpu {
namespace dpcpp {

struct DPCPPGuard {
  explicit DPCPPGuard() = delete;

  explicit DPCPPGuard(DeviceIndex device_index) : guard_(device_index) {}

  explicit DPCPPGuard(Device device) : guard_(device) {}

  DPCPPGuard(const DPCPPGuard&) = delete;
  DPCPPGuard& operator=(const DPCPPGuard&) = delete;

  DPCPPGuard(DPCPPGuard&& other) = delete;
  DPCPPGuard& operator=(DPCPPGuard&& other) = delete;

  void set_device(Device device) {
    guard_.set_device(device);
  }

  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  Device original_device() const {
    return guard_.original_device();
  }

  Device current_device() const {
    return guard_.current_device();
  }

 private:
  c10::impl::InlineDeviceGuard<impl::DPCPPGuardImpl> guard_;
};

struct OptionalDPCPPGuard {
  explicit OptionalDPCPPGuard() : guard_() {}

  explicit OptionalDPCPPGuard(optional<Device> device_opt)
      : guard_(device_opt) {}

  explicit OptionalDPCPPGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  OptionalDPCPPGuard(const OptionalDPCPPGuard&) = delete;
  OptionalDPCPPGuard& operator=(const OptionalDPCPPGuard&) = delete;

  OptionalDPCPPGuard(OptionalDPCPPGuard&& other) = delete;

  OptionalDPCPPGuard& operator=(OptionalDPCPPGuard&& other) = delete;

  void set_device(Device device) {
    guard_.set_device(device);
  }

  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  optional<Device> original_device() const {
    return guard_.original_device();
  }

  optional<Device> current_device() const {
    return guard_.current_device();
  }

  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::DPCPPGuardImpl> guard_;
};

struct DPCPPStreamGuard {
  explicit DPCPPStreamGuard() = delete;

  explicit DPCPPStreamGuard(Stream stream) : guard_(stream) {}

  DPCPPStreamGuard(const DPCPPStreamGuard&) = delete;
  DPCPPStreamGuard& operator=(const DPCPPStreamGuard&) = delete;

  DPCPPStreamGuard(DPCPPStreamGuard&& other) = delete;
  DPCPPStreamGuard& operator=(DPCPPStreamGuard&& other) = delete;

  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  DPCPPStream original_stream() const {
    return DPCPPStream(DPCPPStream::UNCHECKED, guard_.original_stream());
  }

  DPCPPStream current_stream() const {
    return DPCPPStream(DPCPPStream::UNCHECKED, guard_.current_stream());
  }

  Device current_device() const {
    return guard_.current_device();
  }

  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::DPCPPGuardImpl> guard_;
};

struct OptionalDPCPPStreamGuard {
  explicit OptionalDPCPPStreamGuard() : guard_() {}

  explicit OptionalDPCPPStreamGuard(Stream stream) : guard_(stream) {}

  explicit OptionalDPCPPStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  OptionalDPCPPStreamGuard(const OptionalDPCPPStreamGuard&) = delete;
  OptionalDPCPPStreamGuard& operator=(const OptionalDPCPPStreamGuard&) = delete;

  OptionalDPCPPStreamGuard(OptionalDPCPPStreamGuard&& other) = delete;

  OptionalDPCPPStreamGuard& operator=(OptionalDPCPPStreamGuard&& other) =
      delete;

  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  optional<DPCPPStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(DPCPPStream(DPCPPStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  optional<DPCPPStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(DPCPPStream(DPCPPStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::DPCPPGuardImpl> guard_;
};
}
}
