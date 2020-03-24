#include <ATen/Config.h>
#include <ATen/Context.h>
#include <c10/util/Exception.h>

#include <core/DPCPPUtils.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <core/detail/Hooks.h>
#include <core/detail/HooksInterface.h>
#include <utils/General.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace dpcpp {
namespace detail {

void DPCPPHooks::initDPCPP() const {
  // TODO:
  // global state is removed
}

Generator *
DPCPPHooks::getDefaultDPCPPGenerator(DeviceIndex device_index) const {
  return at::dpcpp::detail::getDefaultDPCPPGenerator(device_index);
}

Device DPCPPHooks::getDeviceFromPtr(void *data) const {
  return getDeviceFromPtr(data);
}

bool DPCPPHooks::hasDPCPP() const {
  int count;
  dpcppGetDeviceCount(&count);
  return true;
}

int64_t DPCPPHooks::current_device() const {
  c10::DeviceIndex device;
  dpcppGetDevice(&device);
  return device;
}

int DPCPPHooks::getNumGPUs() const {
  int count;
  dpcppGetDeviceCount(&count);
  return count;
}

bool DPCPPHooks::compiledWithSyCL() const {
#ifndef USE_DPCPP
  return false;
#else
  return true;
#endif
}

// Sigh, the registry doesn't su[[prt namespace :(
using at::DPCPPHooksRegistry;
using at::RegistererDPCPPHooksRegistry;

REGISTER_DPCPP_HOOKS(DPCPPHooks);

} // detail
} // dpcpp
} // namespace
