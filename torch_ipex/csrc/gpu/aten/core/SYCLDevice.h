#ifndef SYCL_DEVICE_INC
#define SYCL_DEVICE_INC

#include <vector>
#include <mutex>

#include <CL/sycl.hpp>

#include <c10/core/Device.h>
#include <core/SYCLException.h>
#include <core/SYCLUtils.h>


namespace c10 { namespace sycl {

struct SYCLDevicePool {
  SYCLDevicePool(): cur_dev_index(-1) {}
  std::vector<cl::sycl::device> devices;
  std::mutex devices_mutex;
  DeviceIndex cur_dev_index;
};

inline Device getDeviceFromPtr(void* ptr) {
  c10::DeviceIndex  device;
  C10_SYCL_CHECK(c10::sycl::syclGetDeviceIdFromPtr(&device, ptr));
  return {DeviceType::DPCPP, static_cast<int16_t>(device)};
}

} // namespace sycl
} // namespace c10
#endif
