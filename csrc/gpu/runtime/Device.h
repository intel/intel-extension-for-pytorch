#pragma once

#include <core/DeviceInfo.h>
#include <runtime/DeviceProp.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

using DeviceId = at::DeviceIndex;

int dpcppPrefetchDeviceCount() noexcept;

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(DeviceId* pDI);

int dpcppSetDevice(DeviceId device_id);

bool dpcppIsDevPoolInit();

int dpcppGetDeviceIdFromPtr(DeviceId* device_id, void* ptr);

sycl::device dpcppGetRawDevice(DeviceId device_id);

DeviceProp* dpcppGetCurrentDeviceProperties();

DeviceProp* dpcppGetDeviceProperties(DeviceId device_id = -1);

DeviceInfo* dpcppGetCurrentDeviceInfo();

DeviceInfo* dpcppGetDeviceInfo(DeviceId device_id = -1);

sycl::context dpcppGetDeviceContext(DeviceId device_id = -1);

DeviceId dpcppGetDeviceIndex(sycl::device device);

std::vector<int>& dpcppGetDeviceIdListForCard(int card_id = -1);

std::vector<int> dpcppPrefetchDeviceIdListForCard(int card_id = -1);
} // namespace dpcpp
} // namespace xpu
