#pragma once

#include <c10/xpu/XPUFunctions.h>
#include <runtime/Exception.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace torch_ipex::xpu {
namespace dpcpp {

using DeviceId = at::DeviceIndex;

bool dpcppGetDeviceHasXMX(DeviceId device_id = 0) noexcept;

bool dpcppGetDeviceHas2DBlock(DeviceId device_id = 0) noexcept;

} // namespace dpcpp
} // namespace torch_ipex::xpu
