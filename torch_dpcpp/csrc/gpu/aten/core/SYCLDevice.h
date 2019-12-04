#pragma once

#include <c10/dpcpp/SYCLException.h>
#include <c10/dpcpp/SYCLUtils.h>
namespace at {
namespace sycl {

inline Device getDeviceFromPtr(void* ptr) {
  c10::DeviceIndex  device;
  C10_SYCL_CHECK(c10::sycl::syclGetDeviceIdFromPtr(&device, ptr));
  return {DeviceType::SYCL, static_cast<int16_t>(device)};

}
}
}
