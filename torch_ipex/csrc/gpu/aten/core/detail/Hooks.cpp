#include <ATen/Config.h>
#include <ATen/Context.h>
#include <c10/util/Exception.h>

#include <core/DPCPPUtils.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <core/detail/Hooks.h>
#include <utils/General.h>
#include <core/CachingHostAllocator.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace dpcpp {
namespace detail {

#ifdef USE_USM
void DPCPPHooks::initDPCPP() const {
  // TODO:
}

bool DPCPPHooks::hasDPCPP() const {
  return true;
}

bool DPCPPHooks::hasOneMKL() const {
#ifdef USE_ONEMKL
  return true;
#else
  return false;
#endif
}

bool DPCPPHooks::hasOneDNN() const {
  return true;
}

std::string DPCPPHooks::showConfig() const {
  return "DPCPP backend version: 1.0";
}

int64_t DPCPPHooks::getCurrentDevice() const {
  c10::DeviceIndex device_index;
  dpcppGetDevice(&device_index);
  return device_index;
}

int DPCPPHooks::getDeviceCount() const {
  int count;
  dpcppGetDeviceCount(&count);
  return count;
}

at::Device DPCPPHooks::getDeviceFromPtr(void* data) const {
  return getDeviceFromPtr(data);
}

bool DPCPPHooks::isPinnedPtr(void* data) const {
  return dpcpp_isAllocatedByCachingHostAllocator(data);
}

at::Allocator* DPCPPHooks::getPinnedMemoryAllocator() const {
  return dpcpp_getCachingHostAllocator();
}

at::Generator* DPCPPHooks::getDefaultDPCPPGenerator(DeviceIndex device_index = -1) const {
  return at::dpcpp::detail::getDefaultDPCPPGenerator(device_index);
}

REGISTER_DPCPP_HOOKS(DPCPPHooks);
#endif

} // detail
} // dpcpp
} // namespace
