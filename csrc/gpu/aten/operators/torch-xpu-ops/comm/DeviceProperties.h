#pragma once

#include <ATen/xpu/XPUContext.h>

#include <iostream>
#include "aten/operators/torch-xpu-ops/comm/Runtime.h"

namespace xpu {
namespace sycl {

template <class KernelClass>
static int64_t syclMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto q = c10::xpu::getCurrentXPUStream(dev_id).queue();
  auto ctx = q.get_context();
  auto dev = q.get_device();

  auto kid = ::sycl::get_kernel_id<KernelClass>();
  // The kernel won't be built for devices except for the first device.
  // Launching kernel on devices except for the first device will raise
  // runtime error. Here is an alternative as a temporary solution to
  // provide an extra hint to SYCL runtime.
  // https://github.com/intel/llvm/issues/15127
  auto kbundle = ::sycl::get_kernel_bundle<::sycl::bundle_state::executable>(
      ctx, {dev}, {kid});

  ::sycl::kernel k = kbundle.get_kernel(kid);
  return k.get_info<::sycl::info::kernel_device_specific::work_group_size>(dev);
}

template <class KernelClass>
static int64_t syclMaxWorkGroupSize(
    KernelClass /*kfn*/,
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  return syclMaxWorkGroupSize<KernelClass>(dev_id);
}

static inline int64_t syclDeviceMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

static inline int64_t syclMaxSubGroupSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto subgroup_sizes = dev_prop->sub_group_sizes;
  uint64_t max_val = 0;
  for (auto i : subgroup_sizes) {
    if (i > max_val)
      max_val = i;
  }
  return max_val;
}

static inline int64_t syclMinSubGroupSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto subgroup_sizes = dev_prop->sub_group_sizes;
  uint64_t min_val = dev_prop->max_work_group_size;
  for (auto i : subgroup_sizes) {
    if (i < min_val)
      min_val = i;
  }
  return min_val;
}

static inline int64_t syclMaxComputeUnitSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_compute_units;
}

static inline int64_t syclGpuEuCount(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_count;
}

static inline int64_t syclGpuEuSimdWidth(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_simd_width;
}

static inline int64_t syclGpuHWThreadsPerEU(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_hw_threads_per_eu;
}

static inline int64_t syclGpuEUCountPerSubslice(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_count_per_subslice;
}

static inline int64_t syclMaxWorkItemsPerTile(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t eu_cnt = dev_prop->gpu_eu_count;
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return eu_cnt * simd_width * hw_threads;
}

static inline int64_t syclMaxWorkItemsPerEU(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return simd_width * hw_threads;
}

static inline int64_t syclMaxDSSNum(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  // TODO: We need to got this info from DPC++ Runtime
  // Hardcode to 32 for ATS
  int64_t dss_num = 32;
  return dss_num;
}

static inline size_t syclGlobalMemSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->global_mem_size;
}

static inline int64_t syclLocalMemSize(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->local_mem_size;
}

template <typename T>
uint32_t syclPrefVectorWidth(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->preferred_vector_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->preferred_vector_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->preferred_vector_width_int;
  }
  if (std::is_same<T, int64_t>::value) {
    return dev_prop->preferred_vector_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->preferred_vector_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->preferred_vector_width_double;
  }
  if (std::is_same<T, ::sycl::half>::value) {
    return dev_prop->preferred_vector_width_half;
  }
  throw std::invalid_argument(
      "Invalid data type to fetch preferred vector width!");
}

template <typename T>
uint32_t syclNativeVectorWidth(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->native_vector_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->native_vector_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->native_vector_width_int;
  }
  if (std::is_same<T, int64_t>::value) {
    return dev_prop->native_vector_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->native_vector_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->native_vector_width_double;
  }
  if (std::is_same<T, ::sycl::half>::value) {
    return dev_prop->native_vector_width_half;
  }
  throw std::invalid_argument(
      "Invalid data type to fetch native vector width!");
}

static inline bool syclHasFloat64(
    at::DeviceIndex dev_id = at::xpu::getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->has_fp64;
}

} // namespace sycl
} // namespace xpu
