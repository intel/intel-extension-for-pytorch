#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <runtime/DPCPP.h>
#include <runtime/Macros.h>

namespace xpu {
namespace dpcpp {

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(at::DeviceIndex* pDI);

int dpcppSetDevice(at::DeviceIndex device_index);

int dpcppGetDeviceIdFromPtr(at::DeviceIndex* device_id, void* ptr);

DPCPP::device dpcppGetRawDevice(at::DeviceIndex device_index);

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
} // namespace at
