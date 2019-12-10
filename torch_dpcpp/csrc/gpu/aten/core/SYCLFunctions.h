#pragma once

// This header provides C++ wrappers around commonly used SYCL API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.dpcpp

// #include <sycl_runtime_api.h>

#include <c10/macros/Macros.h>
#include <c10/core/Device.h>
#include <c10/dpcpp/SYCLException.h>
#include <c10/dpcpp/SYCLUtils.h>

namespace c10 {
namespace sycl {

inline DeviceIndex device_count() noexcept {
  int count;
  // NB: In the past, we were inconsistent about whether or not this reported
  // an error if there were driver problems are not.  Based on experience
  // interacting with users, it seems that people basically ~never want this
  // function to fail; it should just return zero if things are not working.
  // Oblige them.
  int err = syclGetDeviceCount(&count);
  if (err != SYCL_SUCCESS)
    return 0;
  return static_cast<DeviceIndex>(count);
}

inline DeviceIndex current_device() {
  DeviceIndex cur_device;
  C10_SYCL_CHECK(syclGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

inline void set_device(DeviceIndex device) {
  C10_SYCL_CHECK(syclSetDevice(static_cast<int>(device)));
}

}} // namespace c10::sycl
