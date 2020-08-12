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

namespace at {

struct CAFFE2_API DPCPPHooksInterface {
  virtual ~DPCPPHooksInterface() {}

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

struct CAFFE2_API DPCPPHooksArgs {};

C10_DECLARE_REGISTRY(DPCPPHooksRegistry, DPCPPHooksInterface, DPCPPHooksArgs);
#define REGISTER_DPCPP_HOOKS(clsname) \
  C10_REGISTER_CLASS(DPCPPHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const DPCPPHooksInterface& getDPCPPHooks();
} // namespace detail
} // namespace at
