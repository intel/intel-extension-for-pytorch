#include <ATen/Config.h>
#include <ATen/Context.h>
#include <c10/util/Exception.h>

#include <core/Allocator.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <core/detail/Hooks.h>
#include <runtime/Exception.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace xpu {
namespace dpcpp {
namespace detail {

void XPUHooks::initXPU() const {
  // TODO:
}

bool XPUHooks::hasXPU() const {
  return true;
}

bool XPUHooks::hasOneMKL() const {
#ifdef USE_ONEMKL
  return true;
#else
  return false;
#endif
}

bool XPUHooks::hasOneDNN() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "DPCPP backend version: 1.0";
}

int64_t XPUHooks::getCurrentDevice() const {
  return current_device();
}

int XPUHooks::getDeviceCount() const {
  return device_count();
}

at::Device XPUHooks::getDeviceFromPtr(void* data) const {
  auto device = get_device_index_from_ptr(data);
  return {DeviceType::XPU, static_cast<int16_t>(device)};
}

bool XPUHooks::isPinnedPtr(void* data) const {
  return isAllocatedByHostAlloc(data);
}

at::Allocator* XPUHooks::getPinnedMemoryAllocator() const {
  return getHostAllocator();
}

const Generator& XPUHooks::getDefaultXPUGenerator(
    DeviceIndex device_index) const {
  return xpu::dpcpp::detail::getDefaultDPCPPGenerator(device_index);
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace detail
} // namespace dpcpp
} // namespace xpu
