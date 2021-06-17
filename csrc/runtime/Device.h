#pragma once

#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

using DeviceId = int16_t;

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(DeviceId* pDI);

int dpcppSetDevice(DeviceId device_id);

int dpcppGetDeviceIdFromPtr(DeviceId* device_id, void* ptr);

DPCPP::device dpcppGetRawDevice(DeviceId device_id);

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
XPUDeviceProp* getDeviceProperties(DeviceId device_id);

} // namespace dpcpp
} // namespace at
