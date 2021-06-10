#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <utils/DPCPP.h>

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

int dpcppGetDevice(at::DeviceIndex* pDI);

int dpcppSetDevice(at::DeviceIndex device_index);

int dpcppGetDeviceIdFromPtr(at::DeviceIndex* device_id, void* ptr);

DPCPP::device dpcppGetRawDevice(at::DeviceIndex device_index);

DeviceSelector dpcppGetDeviceSelector(at::DeviceIndex device_index);

} // namespace dpcpp
} // namespace at
