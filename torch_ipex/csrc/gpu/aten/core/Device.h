#ifndef DPCPP_DEVICE_INC
#define DPCPP_DEVICE_INC

#include <c10/core/Device.h>
#include <core/Exception.h>

#include <core/DPCPP.h>

#include <vector>
#include <mutex>


namespace at { namespace dpcpp {

class DPCPPDeviceSelector : public DPCPP::device_selector {
public:
  DPCPPDeviceSelector(const DPCPP::device &dev) :
      m_target_device(dev.get()) {}

  DPCPPDeviceSelector(const DPCPPDeviceSelector &other) :
      m_target_device(other.get_target_device()) {}

  int operator()(const DPCPP::device& candidate) const override {
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

struct DPCPPDevicePool {
  DPCPPDevicePool(): cur_dev_index(-1) {}
  std::vector<DPCPP::device> devices;
  std::vector<DPCPPDeviceSelector> dev_sels;
  std::mutex devices_mutex;
  DeviceIndex cur_dev_index;
};

int dpcppGetDeviceIdFromPtr(DeviceIndex *device_id, void *ptr);

inline Device getDeviceFromPtr(void* ptr) {
  c10::DeviceIndex  device;
  AT_DPCPP_CHECK(dpcppGetDeviceIdFromPtr(&device, ptr));
  return {DeviceType::DPCPP, static_cast<int16_t>(device)};
}

} // namespace dpcpp
} // namespace at
#endif
