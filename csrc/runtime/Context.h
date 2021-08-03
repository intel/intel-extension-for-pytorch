#pragma once

#include <runtime/Device.h>
#include <utils/DPCPP.h>

namespace xpu {
namespace dpcpp {

void clearDeviceContext();

DPCPP::context getDeviceContext(DeviceId device_id = 0);

} // namespace dpcpp
} // namespace xpu
