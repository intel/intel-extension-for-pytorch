#ifndef DPCPP_DEVICE_INC
#define DPCPP_DEVICE_INC

#include <c10/core/Device.h>
#include <core/Exception.h>

#include <core/DPCPP.h>

#include <mutex>
#include <vector>

namespace at {
namespace dpcpp {

class DPCPPDeviceSelector : public DPCPP::device_selector {
 public:
  DPCPPDeviceSelector(const DPCPP::device& dev) : m_device(dev) {}

  DPCPPDeviceSelector(const DPCPPDeviceSelector& other)
      : DPCPP::device_selector(other),
        m_device(other.m_device) {}

  int operator()(const DPCPP::device& candidate) const override {
    if (candidate.is_gpu() && candidate == m_device)
      return 100;
    else
      return -1;
  }

 private:
   const DPCPP::device& m_device;
};

struct DPCPPDevicePool {
  std::vector<DPCPP::device> devices;
  std::vector<DPCPPDeviceSelector> dev_sels;
  std::mutex devices_mutex;
};

int dpcppGetDeviceIdFromPtr(DeviceIndex* device_id, void* ptr);

inline Device getDeviceFromPtr(void* ptr) {
  c10::DeviceIndex device;
  AT_DPCPP_CHECK(dpcppGetDeviceIdFromPtr(&device, ptr));
  return {DeviceType::XPU, static_cast<int16_t>(device)};
}

} // namespace dpcpp
} // namespace at
#endif
