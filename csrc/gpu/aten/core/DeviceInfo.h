#pragma once

#include <string>

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
  uint64_t global_mem_size;
  uint32_t max_compute_units;
  bool support_fp64;
};

} // namespace dpcpp
} // namespace xpu
