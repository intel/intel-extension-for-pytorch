#include <core/detail/HooksInterface.h>

#include <ATen/Generator.h>

namespace at { namespace dpcpp { namespace detail {

// The real implementation of DPCPPHooksInterface
struct DPCPPHooks : public at::DPCPPHooksInterface {
  DPCPPHooks(at::DPCPPHooksArgs) {}
  void initDPCPP() const override;
  Device getDeviceFromPtr(void* data) const override;
  Generator* getDefaultDPCPPGenerator(DeviceIndex device_index = -1) const override;
  bool hasDPCPP() const override;
  // bool hasMAGMA() const override;
  // bool hasSyCL() const override;
  // const at::dpcpp::NVRTC& nvrtc() const override;
  int64_t current_device() const override;
  // bool hasPrimaryContext(int64_t device_index) const override;
  // Allocator* getPinnedMemoryAllocator() const override;
  bool compiledWithSyCL() const override;
  // bool compiledWithMIOpen() const override;
  // bool supportsDilatedConvolutionWithSyCL() const override;
  // bool supportsDepthwiseConvolutionWithSyCL() const override;
  // long versionSyCL() const override;
  // std::string showConfig() const override;
  // double batchnormMinEpsilonSyCL() const override;
  // int64_t dpcppFFTGetPlanCacheMaxSize(int64_t device_index) const override;
  // void dpcppFFTSetPlanCacheMaxSize(int64_t device_index, int64_t max_size) const override;
  // int64_t dpcppFFTGetPlanCacheSize(int64_t device_index) const override;
  // void dpcppFFTClearPlanCache(int64_t device_index) const override;
  int getNumGPUs() const override;
};

}}} // at::dpcpp::detail
