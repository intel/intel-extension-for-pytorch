#pragma once

#include <utils/DPCPP.h>
#include <runtime/Device.h>
#include <runtime/Queue.h>


using namespace at;

namespace xpu {
namespace dpcpp {

static inline bool dpcppIsAvailable() {
  int count;
  dpcppGetDeviceCount(&count);
  return count > 0;
}

static inline bool dpcppIsDeviceAvailable(DeviceId dev_id) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->is_available;
}

static inline DPCPP::queue& dpcppGetCurrentQueue() {
  return getCurrentQueue()->getDpcppQueue();
}

static inline DeviceId dpcppGetDeviceIdOfCurrentQueue() {
  return getDeviceIdOfCurrentQueue();
}

static inline int64_t dpcppMaxWorkGroupSize(DeviceId dev_id) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

static inline int64_t dpcppMaxWorkGroupSize() {
  return dpcppMaxWorkGroupSize(getDeviceIdOfCurrentQueue());
}

static inline int64_t dpcppMaxComputeUnitSize(DeviceId dev_id) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->max_compute_units;
}

static inline int64_t dpcppMaxComputeUnitSize() {
  return dpcppMaxComputeUnitSize(getDeviceIdOfCurrentQueue());
}

static inline int64_t dpcppMaxDSSNum(DeviceId dev_id) {
  // TODO: We need to got this info from DPC++ Runtime
  // Hardcode to 32 for ATS
  int64_t dss_num = 32;
  return dss_num;
}

static inline int64_t dpcppMaxDSSNum() {
  return dpcppMaxDSSNum(getDeviceIdOfCurrentQueue());
}

static inline size_t dpcppGlobalMemSize(DeviceId dev_id) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->global_mem_size;
}

static inline size_t dpcppGlobalMemSize() {
  return dpcppGlobalMemSize(getDeviceIdOfCurrentQueue());
}

static inline int64_t dpcppLocalMemSize(DeviceId dev_id) {
  auto* dev_prop = dpcppGetDeviceProperties(dev_id);
  return dev_prop->local_mem_size;
}

static inline int64_t dpcppLocalMemSize() {
  return dpcppLocalMemSize(getDeviceIdOfCurrentQueue());
}


} // namespace dpcpp
} // namespace xpu
