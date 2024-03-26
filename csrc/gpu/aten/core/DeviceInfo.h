#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xpu {
namespace dpcpp {

enum device_type {
  cpu, // Maps to OpenCL CL_DEVICE_TYPE_CPU
  gpu, // Maps to OpenCL CL_DEVICE_TYPE_GPU
  accelerator, // Maps to OpenCL CL_DEVICE_TYPE_ACCELERATOR
  custom, // Maps to OpenCL CL_DEVICE_TYPE_CUSTOM
  automatic, // Maps to OpenCL CL_DEVICE_TYPE_DEFAULT
  host,
  all // Maps to OpenCL CL_DEVICE_TYPE_ALL
};

struct DeviceInfo {
  device_type dev_type;
  std::string dev_name;
  std::string platform_name;
  std::string vendor;
  std::string driver_version;
  std::string version;
  uint32_t device_id;
  uint64_t global_mem_size;
  uint32_t max_compute_units;
  uint32_t gpu_eu_count;
  uint32_t gpu_subslice_count;
  size_t max_work_group_size;
  uint32_t max_num_sub_groups;
  std::vector<size_t> sub_group_sizes;
  bool support_fp64;
};

} // namespace dpcpp
} // namespace xpu
