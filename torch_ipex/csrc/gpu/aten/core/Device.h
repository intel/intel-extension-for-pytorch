#ifndef DPCPP_DEVICE_INC
#define DPCPP_DEVICE_INC

#include <c10/core/Device.h>
#include <core/Exception.h>

#include <core/DPCPP.h>

#include <mutex>
#include <vector>


using namespace at;

namespace xpu {
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

struct XPUDeviceProp {
  typename DPCPP::info::param_traits<DPCPP::info::device, dpcpp_dev_name>::return_type
    name;
  typename DPCPP::info::param_traits<DPCPP::info::device, dpcpp_dev_type>::return_type
    dev_type;
  typename DPCPP::info::param_traits<DPCPP::info::device, dpcpp_dev_global_mem_size>::return_type
    total_global_mem;
  typename DPCPP::info::param_traits<DPCPP::info::device, dpcpp_dev_max_units>::return_type
    max_compute_units;
  typename DPCPP::info::param_traits<DPCPP::info::platform, DPCPP::info::platform::name>::return_type
    platform_name;
  typename DPCPP::info::param_traits<DPCPP::info::device, DPCPP::info::device::partition_max_sub_devices>::return_type
    sub_devices_number;
//  static constexpr auto dpcpp_dev_local_mem_type =
//          DPCPP::info::device::local_mem_type;
//  static constexpr auto dpcpp_dev_local_mem_size =
//          DPCPP::info::device::local_mem_size;
//  static constexpr auto dpcpp_dev_global_mem_size =
//          DPCPP::info::device::global_mem_size;
};

XPUDeviceProp* getCurrentDeviceProperties();
XPUDeviceProp* getDeviceProperties(int64_t device);

} // namespace dpcpp
} // namespace xpu
#endif
