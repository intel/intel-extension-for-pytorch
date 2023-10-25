#pragma once

#include <core/DeviceInfo.h>
#include <runtime/DeviceProp.h>
#include <runtime/Exception.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

using DeviceId = at::DeviceIndex;

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(DeviceId* pDI);

int dpcppSetDevice(DeviceId device_id);

bool dpcppIsDevPoolInit();

int dpcppGetDeviceIdFromPtr(DeviceId* device_id, void* ptr);

sycl::device& dpcppGetRawDevice(DeviceId device_id);

DeviceProp* dpcppGetCurrentDeviceProperties();

DeviceProp* dpcppGetDeviceProperties(DeviceId device_id = -1);

DeviceInfo* dpcppGetCurrentDeviceInfo();

DeviceInfo* dpcppGetDeviceInfo(DeviceId device_id = -1);

sycl::context& dpcppGetDeviceContext(DeviceId device_id = -1);

DeviceId dpcppGetDeviceIndex(sycl::device device);

int dpcppPrefetchDeviceCount(int& device_count) noexcept;

int dpcppPrefetchDeviceHasFP64Dtype(int device_id, bool& has_fp64) noexcept;

} // namespace dpcpp
} // namespace xpu
