#pragma once

#include <c10/core/Device.h>

#include <core/DeviceProp.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

IPEX_API DeviceIndex device_count() noexcept;

IPEX_API DeviceIndex current_device();

IPEX_API void set_device(DeviceIndex device);

DeviceIndex get_devie_index_from_ptr(void* ptr);

IPEX_API DeviceProp* getCurrentDeviceProperties();

IPEX_API DeviceProp* getDeviceProperties(DeviceIndex device);

} // namespace dpcpp
} // namespace xpu
