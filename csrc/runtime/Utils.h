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

static inline DPCPP::queue& dpcppGetCurrentQueue() {
  return getCurrentQueue()->getDpcppQueue();
}

static inline int64_t dpcppMaxWorkGroupSize(DPCPP::queue& queue) {
  return queue.get_device().get_info<dpcpp_dev_max_wgroup_size>();
}

static inline int64_t dpcppMaxWorkGroupSize() {
  auto& queue = dpcppGetCurrentQueue();
  return dpcppMaxWorkGroupSize(queue);
}

static inline int64_t dpcppMaxComputeUnitSize(DPCPP::queue& queue) {
  return queue.get_device()
      .template get_info<dpcpp_dev_max_units>();
}

static inline int64_t dpcppMaxComputeUnitSize() {
  auto& queue = dpcppGetCurrentQueue();
  return dpcppMaxComputeUnitSize(queue);
}

static inline int64_t dpcppMaxDSSNum(DPCPP::queue& queue) {
  // TODO: We need to got this info from DPC++ Runtime
  // Hardcode to 32 for ATS
  int64_t dss_num = 32;
  return dss_num;
}

static inline int64_t dpcppMaxDSSNum() {
  auto& queue = dpcppGetCurrentQueue();
  return dpcppMaxDSSNum(queue);
}

static inline int64_t dpcppLocalMemSize(DPCPP::queue& queue) {
  return queue.get_device()
      .template get_info<dpcpp_dev_local_mem_size>();
}

static inline int64_t dpcppLocalMemSize() {
  auto& queue = dpcppGetCurrentQueue();
  return dpcppLocalMemSize(queue);
}

} // namespace dpcpp
} // namespace xpu
