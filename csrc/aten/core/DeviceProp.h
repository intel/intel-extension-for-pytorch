#pragma once

#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

struct IPEX_API DeviceProp {
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
};

} // namespace dpcpp
} // namespace xpu
