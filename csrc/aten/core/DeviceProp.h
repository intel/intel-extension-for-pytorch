#pragma once

#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

struct IPEX_API DeviceProp {
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_name> dev_name;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_type> dev_type;
  dpcpp_info_t<DPCPP::info::platform, dpcpp_platform_name> platform_name;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_vendor> vendor;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_driver_version> driver_version;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_version> version;
  // dpcpp_info_t<DPCPP::info::device, dpcpp_dev_backend_version>
  // backend_version;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_is_available> is_available;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_param_size> max_param_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_compute_units>
      max_compute_units;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_work_item_dims>
      max_work_item_dims;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_work_group_size>
      max_work_group_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_num_subgroup>
      max_num_subgroup;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_subgroup_sizes> subgroup_sizes;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_clock_freq> max_clock_freq;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_address_bits> address_bits;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_alloc_size>
      max_mem_alloc_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_mem_base_addr_align>
      base_addr_align;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_half_fp_config> half_fp_config;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_single_fp_config>
      single_fp_config;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_double_fp_config>
      double_fp_config;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_global_mem_size> global_mem_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_global_mem_cache_type>
      global_mem_cache_type;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_global_mem_cache_size>
      global_mem_cache_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_global_mem_cache_line_size>
      global_mem_cache_line_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_local_mem_type> local_mem_type;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_local_mem_size> local_mem_size;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_max_sub_devices> max_sub_devices;
  dpcpp_info_t<DPCPP::info::device, dpcpp_dev_profiling_resolution>
      profiling_resolution;
};

} // namespace dpcpp
} // namespace xpu
