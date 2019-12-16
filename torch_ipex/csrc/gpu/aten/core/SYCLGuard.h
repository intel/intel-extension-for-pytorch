#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <core/SYCLGuardImpl.h>
#include <core/SYCLMacros.h>
#include <cstddef>

namespace c10 { namespace sycl {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard boilerplate]

/// A variant of DeviceGuard that is specialized for SYCL.  It accepts
/// integer indices (interpreting them as SYCL devices) and is a little
/// more efficient than DeviceGuard however, it can only be used
/// from code that links against SYCL directly.
struct SYCLGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit SYCLGuard() = delete;

  /// Set the current SYCL device to the passed device index.
  explicit SYCLGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current SYCL device to the passed device.  Errors if the passed
  /// device is not a SYCL device.
  explicit SYCLGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  SYCLGuard(const SYCLGuard&) = delete;
  SYCLGuard& operator=(const SYCLGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  SYCLGuard(SYCLGuard&& other) = delete;
  SYCLGuard& operator=(SYCLGuard&& other) = delete;

  /// Sets the SYCL device to the given device.  Errors if the given device
  /// is not a SYCL device.
  void set_device(Device device) { guard_.set_device(device); }

  /// Sets the SYCL device to the given device.  Errors if the given device
  /// is not a SYCL device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) { guard_.reset_device(device); }

  /// Sets the SYCL device to the given device index.
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const { return guard_.original_device(); }

  /// Returns the last device that was set via `set_device`, if any, otherwise the
  /// device passed during construction.
  Device current_device() const { return guard_.current_device(); }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::SYCLGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for SYCL.  See
/// SYCLGuard for when you can use this.
struct OptionalSYCLGuard {
  /// Create an uninitialized OptionalSYCLGuard.
  explicit OptionalSYCLGuard() : guard_() {}

  /// Set the current SYCL device to the passed Device, if it is not nullopt.
  explicit OptionalSYCLGuard(optional<Device> device_opt) : guard_(device_opt) {}

  /// Set the current SYCL device to the passed device index, if it is not
  /// nullopt
  explicit OptionalSYCLGuard(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalSYCLGuard(const OptionalSYCLGuard&) = delete;
  OptionalSYCLGuard& operator=(const OptionalSYCLGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalSYCLGuard(OptionalSYCLGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalSYCLGuard& operator=(OptionalSYCLGuard&& other) = delete;

  /// Sets the SYCL device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a SYCL device.
  void set_device(Device device) { guard_.set_device(device); }

  /// Sets the SYCL device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a SYCL device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) { guard_.reset_device(device); }

  /// Sets the SYCL device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const { return guard_.original_device(); }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const { return guard_.current_device(); }

  /// Restore the original SYCL device, resetting this guard to uninitialized state.
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalDeviceGuard<impl::SYCLGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for SYCL.  See SYCLGuard
/// for when you can use this.
struct SYCLStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit SYCLStreamGuard() = delete;

  /// Set the current SYCL device to the device associated with the passed stream,
  /// and set the current SYCL stream on that device to the passed stream.
  /// Errors if the Stream is not a SYCL stream.
  explicit SYCLStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  SYCLStreamGuard(const SYCLStreamGuard&) = delete;
  SYCLStreamGuard& operator=(const SYCLStreamGuard&) = delete;

  /// Move is disallowed, as SYCLStreamGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  SYCLStreamGuard(SYCLStreamGuard&& other) = delete;
  SYCLStreamGuard& operator=(SYCLStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a SYCL stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on SYCL, use SYCLMultiStreamGuard instead.
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  /// Returns the SYCL stream that was set at the time the guard was constructed.
  SYCLStream original_stream() const {
    return SYCLStream(SYCLStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent SYCL stream that was set using this device guard,
  /// either from construction, or via set_stream.
  SYCLStream current_stream() const {
    return SYCLStream(SYCLStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent SYCL device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const { return guard_.current_device(); }

  /// Returns the SYCL device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const { return guard_.original_device(); }

private:
  c10::impl::InlineStreamGuard<impl::SYCLGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for SYCL.  See SYCLGuard
/// for when you can use this.
struct OptionalSYCLStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalSYCLStreamGuard() : guard_() {}

  /// Set the current SYCL device to the device associated with the passed stream,
  /// and set the current SYCL stream on that device to the passed stream.
  /// Errors if the Stream is not a SYCL stream.
  explicit OptionalSYCLStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalSYCLStreamGuard(optional<Stream> stream_opt) : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalSYCLStreamGuard(const OptionalSYCLStreamGuard&) = delete;
  OptionalSYCLStreamGuard& operator=(const OptionalSYCLStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalSYCLStreamGuard(OptionalSYCLStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalSYCLStreamGuard& operator=(OptionalSYCLStreamGuard&& other) = delete;

  /// Resets the currently set SYCL stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  /// Returns the SYCL stream that was set at the time the guard was most recently
  /// initialized, or nullopt if the guard is uninitialized.
  optional<SYCLStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(SYCLStream(SYCLStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent SYCL stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<SYCLStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(SYCLStream(SYCLStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original SYCL device and stream, resetting this guard to uninitialized state.
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalStreamGuard<impl::SYCLGuardImpl> guard_;
};
} // namespace sycl
} // namespace c10
