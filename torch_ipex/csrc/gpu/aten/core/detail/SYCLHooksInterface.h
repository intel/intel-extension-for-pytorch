#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

//Forward-declares THSYCLState
struct THSYCLState;

namespace at {
class Context;
}

//NB: Class must live in "at" due to limitations of Registry.h.
namespace at {

// The SYCLHooksInterface is an omnibus interface for any SYCL functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of SYCL code). See
// CUDAHooksInterface for more detailed motivation.
struct CAFFE2_API SYCLHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~SYCLHooksInterface() {}

  // Initialize THSYCLState and, transitively, the SYCL state
  virtual std::unique_ptr<THSYCLState, void (*)(THSYCLState*)> initSYCL() const {
    TORCH_CHECK("Cannot initialize SYCL without ATen_sycl library.");
    return {nullptr, nullptr};
  }

  virtual Generator* getDefaultSYCLGenerator(DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default SYCL generator without ATen_sycl library. ");
  }

  virtual Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK(false, "Cannot get device of pointer on SYCL without ATen_sycl library. ");
  }

  virtual bool hasSYCL() const {
    return false;
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    AT_ERROR("SYCL currently not support Pinned Memory");
  }

  virtual int getNumGPUs() const {
    return 0;
  }

  virtual bool compiledWithSyCL() const {
    return false;
  }
};

//NB: dummy argument to suppress "ISO C++11 requires at least one argument
//for the "..." in a variadic macro"
struct CAFFE2_API SYCLHooksArgs {};

C10_DECLARE_REGISTRY(SYCLHooksRegistry, SYCLHooksInterface, SYCLHooksArgs);
#define REGISTER_SYCL_HOOKS(clsname) \
  C10_REGISTER_CLASS(SYCLHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const SYCLHooksInterface& getSYCLHooks();
} // namespace detail
} // namespace at
