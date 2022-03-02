#pragma once

#include <core/DeviceProp.h>

#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

using DeviceId = int16_t;

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(DeviceId* pDI);

int dpcppSetDevice(DeviceId device_id);

int dpcppGetDeviceIdFromPtr(DeviceId* device_id, void* ptr);

DPCPP::device dpcppGetRawDevice(DeviceId device_id);

DeviceProp* dpcppGetCurrentDeviceProperties();

DeviceProp* dpcppGetDeviceProperties(DeviceId device_id = -1);

DPCPP::context dpcppGetDeviceContext(DeviceId device_id = -1);

} // namespace dpcpp
} // namespace xpu
