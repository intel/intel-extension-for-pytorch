#pragma once


// #include <dpcpp_runtime_api.h>

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <core/DPCPPUtils.h>
#include <core/Exception.h>

namespace at {
namespace dpcpp {

inline DeviceIndex device_count() noexcept {
  int count;
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
} // namespace dpcpp
} // namespace at
