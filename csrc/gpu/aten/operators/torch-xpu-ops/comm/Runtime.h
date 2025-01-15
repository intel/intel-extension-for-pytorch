#pragma once

#include <c10/xpu/XPUStream.h>

namespace at::xpu {

static inline at::DeviceIndex getDeviceIndexOfCurrentQueue() {
  return c10::xpu::getCurrentXPUStream().device_index();
}

static inline sycl::queue& getCurrentSYCLQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

} // namespace at::xpu
