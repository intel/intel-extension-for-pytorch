#include <ATen/detail/DPCPPHooksInterface.h>
#include <ATen/Generator.h>

namespace at { namespace dpcpp { namespace detail {

// The real implementation of DPCPPHooksInterface
struct DPCPPHooks : public at::DPCPPHooksInterface {
  DPCPPHooks(at::DPCPPHooksArgs) {}
  void initDPCPP() const override;
  bool hasDPCPP() const override;
  bool hasOneMKL() const override;
  bool hasOneDNN() const override;
  std::string showConfig() const override;
  int64_t getCurrentDevice() const override;
  int getDeviceCount() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(void* data) const override;
  Allocator* getPinnedMemoryAllocator() const override;
  Generator* getDefaultDPCPPGenerator(DeviceIndex device_index) const override;
};

}}} // at::dpcpp::detail
