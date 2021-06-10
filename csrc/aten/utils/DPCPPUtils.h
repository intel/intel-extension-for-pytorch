#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <utils/DPCPP.h>


using namespace at;

namespace xpu {
namespace dpcpp {

class DeviceSelector : public DPCPP::device_selector {
 public:
  DeviceSelector(const DPCPP::device& dev) : m_device(dev) {}

  DeviceSelector(const DeviceSelector& other)
      : DPCPP::device_selector(other),
        m_device(other.m_device) {}

  int operator()(const DPCPP::device& candidate) const override {
    if (candidate.is_gpu() && candidate == m_device)
      return 100;
    else
      return -1;
  }

 private:
   const DPCPP::device& m_device;
};

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(DeviceIndex* pDI);

int dpcppSetDevice(DeviceIndex device_index);

int dpcppGetDeviceIdFromPtr(DeviceIndex* device_id, void* ptr);

DPCPP::device dpcppGetRawDevice(DeviceIndex device_index);

DeviceSelector dpcppGetDeviceSelector(DeviceIndex device_index);

DPCPP::queue& dpcppGetCurrentQueue();

int64_t dpcppMaxWorkGroupSize();

int64_t dpcppMaxWorkGroupSize(DPCPP::queue& queue);

int64_t dpcppMaxComputeUnitSize();

int64_t dpcppMaxComputeUnitSize(DPCPP::queue& queue);

int64_t dpcppMaxDSSNum();

int64_t dpcppMaxDSSNum(DPCPP::queue& queue);

int64_t dpcppLocalMemSize();

int64_t dpcppLocalMemSize(DPCPP::queue& queue);

static inline bool dpcppIsAvailable() {
  int count;
  dpcppGetDeviceCount(&count);
  return count > 0;
}

} // namespace dpcpp
} // namespace xpu
