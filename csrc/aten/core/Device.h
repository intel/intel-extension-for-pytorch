#pragma once

#include <c10/core/Device.h>

#include <core/DeviceProp.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

DeviceIndex prefetch_device_count() noexcept;

DeviceIndex device_count() noexcept;

DeviceIndex current_device();

void set_device(DeviceIndex device);

DeviceIndex get_device_index_from_ptr(void* ptr);

DeviceProp* getCurrentDeviceProperties();

DeviceProp* getDeviceProperties(DeviceIndex device);

std::vector<int> prefetchDeviceIdListForCard(int card_id);

std::vector<int>& getDeviceIdListForCard(int card_id);
} // namespace dpcpp
} // namespace xpu
