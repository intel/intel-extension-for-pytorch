#pragma once

// This header provides C++ wrappers around commonly used DPCPP API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.dpcpp

// #include <dpcpp_runtime_api.h>

#include <c10/macros/Macros.h>
#include <c10/core/Device.h>
#include <core/Exception.h>
#include <core/DPCPPUtils.h>

namespace at {
namespace dpcpp {

inline DeviceIndex device_count() noexcept {
  int count;
  // NB: In the past, we were inconsistent about whether or not this reported
  // an error if there were driver problems are not.  Based on experience
  // interacting with users, it seems that people basically ~never want this
  // function to fail; it should just return zero if things are not working.
  // Oblige them.
  int err = dpcppGetDeviceCount(&count);
  if (err != DPCPP_SUCCESS)
    return 0;
  return static_cast<DeviceIndex>(count);
}

inline DeviceIndex current_device() {
  DeviceIndex cur_device;
  AT_DPCPP_CHECK(dpcppGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

inline void set_device(DeviceIndex device) {
  AT_DPCPP_CHECK(dpcppSetDevice(static_cast<int>(device)));
}

}} // namespace at::dpcpp
