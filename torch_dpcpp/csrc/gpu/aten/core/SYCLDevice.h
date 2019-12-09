#ifndef SYCL_DEVICE_INC
#define SYCL_DEVICE_INC

#include <vector>
#include <mutex>

#include <CL/sycl.hpp>

#include <c10/core/Device.h>

namespace c10 { namespace sycl {

struct SYCLDevicePool {
  SYCLDevicePool(): cur_dev_index(-1) {}
  std::vector<cl::sycl::device> devices;
  std::mutex devices_mutex;
  DeviceIndex cur_dev_index;
};

} // namespace sycl
} // namespace c10
#endif
