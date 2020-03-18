#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <core/GuardImpl.h>
#include <core/Macros.h>
#include <cstddef>

namespace at {
namespace dpcpp {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for DPCPP.  It accepts
/// integer indices (interpreting them as DPCPP devices) and is a little
/// more efficient than DeviceGuard however, it can only be used
/// from code that links against DPCPP directly.
struct DPCPPGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit DPCPPGuard() = delete;

  /// Set the current DPCPP device to the passed device index.
  explicit DPCPPGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current DPCPP device to the passed device.  Errors if the passed
  /// device is not a DPCPP device.
  explicit DPCPPGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  DPCPPGuard(const DPCPPGuard &) = delete;
  DPCPPGuard &operator=(const DPCPPGuard &) = delete;

  // Move is not allowed (there is no uninitialized state)
  DPCPPGuard(DPCPPGuard &&other) = delete;
  DPCPPGuard &operator=(DPCPPGuard &&other) = delete;

  /// Sets the DPCPP device to the given device.  Errors if the given device
  /// is not a DPCPP device.
  void set_device(Device device) { guard_.set_device(device); }

  /// Sets the DPCPP device to the given device.  Errors if the given device
  /// is not a DPCPP device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) { guard_.reset_device(device); }

  /// Sets the DPCPP device to the given device index.
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const { return guard_.original_device(); }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the
  /// device passed during construction.
  Device current_device() const { return guard_.current_device(); }

private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::DPCPPGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for DPCPP.  See
/// DPCPPGuard for when you can use this.
struct OptionalDPCPPGuard {
  /// Create an uninitialized OptionalDPCPPGuard.
  explicit OptionalDPCPPGuard() : guard_() {}

  /// Set the current DPCPP device to the passed Device, if it is not nullopt.
  explicit OptionalDPCPPGuard(optional<Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current DPCPP device to the passed device index, if it is not
  /// nullopt
  explicit OptionalDPCPPGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalDPCPPGuard(const OptionalDPCPPGuard &) = delete;
  OptionalDPCPPGuard &operator=(const OptionalDPCPPGuard &) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalDPCPPGuard(OptionalDPCPPGuard &&other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalDPCPPGuard &operator=(OptionalDPCPPGuard &&other) = delete;

  /// Sets the DPCPP device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a DPCPP
  /// device.
  void set_device(Device device) { guard_.set_device(device); }

  /// Sets the DPCPP device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a DPCPP
  /// device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) { guard_.reset_device(device); }

  /// Sets the DPCPP device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const { return guard_.original_device(); }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const { return guard_.current_device(); }

  /// Restore the original DPCPP device, resetting this guard to uninitialized
  /// state.
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalDeviceGuard<impl::DPCPPGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for DPCPP.  See DPCPPGuard
/// for when you can use this.
struct DPCPPStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit DPCPPStreamGuard() = delete;

  /// Set the current DPCPP device to the device associated with the passed
  /// stream,
  /// and set the current DPCPP stream on that device to the passed stream.
  /// Errors if the Stream is not a DPCPP stream.
  explicit DPCPPStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  DPCPPStreamGuard(const DPCPPStreamGuard &) = delete;
  DPCPPStreamGuard &operator=(const DPCPPStreamGuard &) = delete;

  /// Move is disallowed, as DPCPPStreamGuard does not have an uninitialized
  /// state,
  /// which is required for moves on types with nontrivial destructors.
  DPCPPStreamGuard(DPCPPStreamGuard &&other) = delete;
  DPCPPStreamGuard &operator=(DPCPPStreamGuard &&other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a DPCPP stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on DPCPP, use DPCPPMultiStreamGuard instead.
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  /// Returns the DPCPP stream that was set at the time the guard was
  /// constructed.
  DPCPPStream original_stream() const {
    return DPCPPStream(DPCPPStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent DPCPP stream that was set using this device guard,
  /// either from construction, or via set_stream.
  DPCPPStream current_stream() const {
    return DPCPPStream(DPCPPStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent DPCPP device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const { return guard_.current_device(); }

  /// Returns the DPCPP device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const { return guard_.original_device(); }

private:
  c10::impl::InlineStreamGuard<impl::DPCPPGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for DPCPP.  See
/// DPCPPGuard
/// for when you can use this.
struct OptionalDPCPPStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalDPCPPStreamGuard() : guard_() {}

  /// Set the current DPCPP device to the device associated with the passed
  /// stream,
  /// and set the current DPCPP stream on that device to the passed stream.
  /// Errors if the Stream is not a DPCPP stream.
  explicit OptionalDPCPPStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalDPCPPStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalDPCPPStreamGuard(const OptionalDPCPPStreamGuard &) = delete;
  OptionalDPCPPStreamGuard &
  operator=(const OptionalDPCPPStreamGuard &) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalDPCPPStreamGuard(OptionalDPCPPStreamGuard &&other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalDPCPPStreamGuard &
  operator=(OptionalDPCPPStreamGuard &&other) = delete;

  /// Resets the currently set DPCPP stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  /// Returns the DPCPP stream that was set at the time the guard was most
  /// recently
  /// initialized, or nullopt if the guard is uninitialized.
  optional<DPCPPStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(DPCPPStream(DPCPPStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent DPCPP stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized,
  /// or nullopt if the guard is uninitialized.
  optional<DPCPPStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(DPCPPStream(DPCPPStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original DPCPP device and stream, resetting this guard to
  /// uninitialized state.
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalStreamGuard<impl::DPCPPGuardImpl> guard_;
};
} // namespace dpcpp
} // namespace at
