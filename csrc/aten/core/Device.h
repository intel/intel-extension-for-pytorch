#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <utils/Exception.h>
#include <utils/DPCPPUtils.h>

#include <mutex>
#include <vector>


using namespace at;

namespace xpu {
namespace dpcpp {



struct DPCPPDevicePool {
  std::vector<DPCPP::device> devices;
  std::vector<DeviceSelector> dev_sels;
  std::mutex devices_mutex;
};

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
} // namespace xpu
