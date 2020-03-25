#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

// Forward-declares THDPCPPState
// struct THDPCPPState;

namespace at {
class Context;
}

//NB: Class must live in "at" due to limitations of Registry.h.
namespace at {

// The DPCPPHooksInterface is an omnibus interface for any DPCPP functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of DPCPP code). See
// CUDAHooksInterface for more detailed motivation.
struct CAFFE2_API DPCPPHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~DPCPPHooksInterface() {}

  // Initialize THDPCPPState and, transitively, the DPCPP state
  // virtual std::unique_ptr<THDPCPPState, void (*)(THDPCPPState*)> initDPCPP() const {
  //   TORCH_CHECK("Cannot initialize DPCPP without ATen_dpcpp library.");
  //   return {nullptr, nullptr};
  // }

  virtual void initDPCPP() const {
    TORCH_CHECK("Cannot initialize DPCPP without ATen_dpcpp library.");
  }

  virtual Generator* getDefaultDPCPPGenerator(DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default DPCPP generator without ATen_dpcpp library. ");
  }

  virtual Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK(false, "Cannot get device of pointer on DPCPP without ATen_dpcpp library. ");
  }

  virtual bool hasDPCPP() const {
    return false;
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(0, "DPCPP currently not support Pinned Memory");
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
struct CAFFE2_API DPCPPHooksArgs {};

C10_DECLARE_REGISTRY(DPCPPHooksRegistry, DPCPPHooksInterface, DPCPPHooksArgs);
#define REGISTER_DPCPP_HOOKS(clsname) \
  C10_REGISTER_CLASS(DPCPPHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const DPCPPHooksInterface& getDPCPPHooks();
} // namespace detail
} // namespace at
