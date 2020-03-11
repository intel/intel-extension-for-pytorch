#include <core/detail/SYCLHooks.h>

#include <core/SYCLGenerator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <core/SYCLDevice.h>
#include <core/detail/SYCLHooksInterface.h>
#include <c10/util/Exception.h>

#include <core/SYCLUtils.h>
#include <utils/General.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace sycl {
namespace detail {

void SYCLHooks::initSYCL() const {
  // TODO:
  // global state is removed
}

Generator* SYCLHooks::getDefaultSYCLGenerator(DeviceIndex device_index) const {
  return at::sycl::detail::getDefaultSYCLGenerator(device_index);
}

Device SYCLHooks::getDeviceFromPtr(void* data) const {
  return c10::sycl::getDeviceFromPtr(data);
}

bool SYCLHooks::hasSYCL() const {
  int count;
  c10::sycl::syclGetDeviceCount(&count);
  return true;
}

int64_t SYCLHooks::current_device() const {
  c10::DeviceIndex  device;
  c10::sycl::syclGetDevice(&device);
  return device;
}

int SYCLHooks::getNumGPUs() const {
  int count;
  c10::sycl::syclGetDeviceCount(&count);
  return count;
}

bool SYCLHooks::compiledWithSyCL() const {
#ifndef USE_SYCL
  return false;
#else
  return true;
#endif
}

// Sigh, the registry doesn't su[[prt namespace :(
using at::SYCLHooksRegistry;
using at::RegistererSYCLHooksRegistry;

REGISTER_SYCL_HOOKS(SYCLHooks);

} // detail
} // sycl
} // namespace
