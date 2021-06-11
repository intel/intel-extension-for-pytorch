#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <utils/Exception.h>
#include <utils/DPCPPUtils.h>

using namespace at;

namespace xpu {
namespace dpcpp {

inline DeviceIndex device_count() noexcept {
  int count;
  int err = dpcppGetDeviceCount(&count);
  return (err == DPCPP_SUCCESS) ? static_cast<DeviceIndex>(count) : 0;
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
} // namespace xpu
