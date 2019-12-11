#include <core/detail/SYCLHooks.h>

#include <ATen/SYCLGenerator.h>
#include <ATen/Context.h>
//#include <core/SYCLConfig.h>
#include <ATen/Config.h>
#include <core/SYCLDevice.h>
#include <ATen/detail/SYCLHooksInterface.h>
#include <c10/util/Exception.h>

#include <legacy/THSYCL.h>

#include <core/SYCLUtils.h>
#include <utils/General.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace sycl {
namespace detail {

std::unique_ptr<THSYCLState, void (*)(THSYCLState*)> SYCLHooks::initSYCL() const {
  THSYCLState* thsycl_state = THSYCLState_alloc();

  THSyclInit(thsycl_state);
  return std::unique_ptr<THSYCLState, void (*)(THSYCLState*)>(
    thsycl_state, [](THSYCLState* p) {
      if (p)
        THSYCLState_free(p);
    });
}

Generator* SYCLHooks::getDefaultSYCLGenerator(DeviceIndex device_index) const {
  return at::sycl::detail::getDefaultSYCLGenerator(device_index);
}

Device SYCLHooks::getDeviceFromPtr(void* data) const {
  return at::sycl::getDeviceFromPtr(data);
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
  return AT_SYCL_ENABLED();
}

// Sigh, the registry doesn't su[[prt namespace :(
using at::SYCLHooksRegistry;
using at::RegistererSYCLHooksRegistry;

REGISTER_SYCL_HOOKS(SYCLHooks);

} // detail
} // sycl
} // namespace
