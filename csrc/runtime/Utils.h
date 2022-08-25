#pragma once

#include <runtime/Device.h>
#include <runtime/Queue.h>
#include <utils/DPCPP.h>
#include <stdexcept>
#include <type_traits>

#define SYCL_MAX_SUB_GROUP_SIZE dpcppMaxSubGroupSize()

using namespace at;

namespace xpu {
namespace dpcpp {

// C++ API for lazy initialization. Note: don't put this function in destructor,
// because it may result in a deadlock.
void lazy_init();

void set_run_yet_variable_to_false();

static inline bool dpcppIsAvailable() {
  int count;
  dpcppGetDeviceCount(&count);
  return count > 0;
}

static inline bool dpcppIsDeviceAvailable(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->is_available;
}

static inline sycl::queue& dpcppGetCurrentQueue() {
  return getCurrentQueue()->getDpcppQueue();
}

static inline DeviceId dpcppGetDeviceIdOfCurrentQueue() {
  return getDeviceIdOfCurrentQueue();
}

static inline QueueId dpcppGetCurrentQueueId() {
  return getQueueId(getCurrentQueue());
}

static inline int64_t dpcppMaxWorkGroupSize(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

static inline int64_t dpcppMaxSubGroupSize(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  auto subgroup_sizes = dev_prop->subgroup_sizes;
  int64_t max_val = 0;
  for (auto i : subgroup_sizes) {
    if (i > max_val)
      max_val = i;
  }
  return max_val;
}

static inline int64_t dpcppMaxComputeUnitSize(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->max_compute_units;
}

static inline int64_t dpcppGpuEuSimdWidth(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->gpu_eu_simd_width;
}

static inline int64_t dpcppGpuHWThreadsPerEU(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->gpu_hw_threads_per_eu;
}

static inline bool dpcppSupportAtomic64(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->support_atomic64;
}

static inline bool dpcppSupportFP64(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->support_fp64;
}

static inline int64_t dpcppMaxWorkItemsPerTile(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  int64_t eu_num = dev_prop->max_compute_units;
  int64_t simd_width = dpcppMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return eu_num * simd_width * hw_threads;
}

static inline int64_t dpcppMaxDSSNum(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  // TODO: We need to got this info from DPC++ Runtime
  // Hardcode to 32 for ATS
  int64_t dss_num = 32;
  return dss_num;
}

static inline size_t dpcppGlobalMemSize(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->global_mem_size;
}

static inline int64_t dpcppLocalMemSize(
    DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->local_mem_size;
}

template <typename T>
uint32_t dpcppPrefVectorWidth(DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->pref_vec_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->pref_vec_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->pref_vec_width_int;
  }
  if (std::is_same<T, long>::value) {
    return dev_prop->pref_vec_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->pref_vec_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->pref_vec_width_double;
  }
  if (std::is_same<T, sycl::half>::value) {
    return dev_prop->pref_vec_width_half;
  }
  throw std::invalid_argument(
      "Invalid data type to fetch preferred vector width!");
}

template <typename T>
uint32_t dpcppNativeVectorWidth(DeviceId dev_id = getDeviceIdOfCurrentQueue()) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->native_vec_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->native_vec_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->native_vec_width_int;
  }
  if (std::is_same<T, long>::value) {
    return dev_prop->native_vec_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->native_vec_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->native_vec_width_double;
  }
  if (std::is_same<T, sycl::half>::value) {
    return dev_prop->native_vec_width_half;
  }
  throw std::invalid_argument(
      "Invalid data type to fetch native vector width!");
}

} // namespace dpcpp
} // namespace xpu
