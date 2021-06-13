#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <runtime/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

IPEX_API DeviceIndex device_count() noexcept;

IPEX_API DeviceIndex current_device();

IPEX_API void set_device(DeviceIndex device);

} // namespace dpcpp
} // namespace xpu
