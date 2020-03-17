#ifndef SYCL_DEVICE_INC
#define SYCL_DEVICE_INC

#include <vector>
#include <mutex>

#include <CL/sycl.hpp>

#include <c10/core/Device.h>
#include <core/Exception.h>


namespace c10 { namespace sycl {

class DPCPPDeviceSelector : public cl::sycl::device_selector {
public:
  DPCPPDeviceSelector(const cl::sycl::device &dev) :
      m_target_device(dev.get()) {}

  DPCPPDeviceSelector(const DPCPPDeviceSelector &other) :
      m_target_device(other.get_target_device()) {}

  int operator()(const cl::sycl::device& candidate) const override {
    if (candidate.is_gpu() && candidate.get() == m_target_device)
      return 100;
    else
      return -1;
  }

  cl_device_id get_target_device() const {
    return m_target_device;
  }

private:
  cl_device_id m_target_device;
};

struct SYCLDevicePool {
  SYCLDevicePool(): cur_dev_index(-1) {}
  std::vector<cl::sycl::device> devices;
  std::vector<DPCPPDeviceSelector> dev_sels;
  std::mutex devices_mutex;
  DeviceIndex cur_dev_index;
};

int syclGetDeviceIdFromPtr(DeviceIndex *device_id, void *ptr);

inline Device getDeviceFromPtr(void* ptr) {
  c10::DeviceIndex  device;
  C10_SYCL_CHECK(syclGetDeviceIdFromPtr(&device, ptr));
  return {DeviceType::DPCPP, static_cast<int16_t>(device)};
}

} // namespace sycl
} // namespace c10
#endif
