#include <core/detail/HooksInterface.h>

#include <ATen/Generator.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at { namespace sycl { namespace detail {

// The real implementation of SYCLHooksInterface
struct SYCLHooks : public at::SYCLHooksInterface {
  SYCLHooks(at::SYCLHooksArgs) {}
  void initSYCL() const override;
  Device getDeviceFromPtr(void* data) const override;
  Generator* getDefaultSYCLGenerator(DeviceIndex device_index = -1) const override;
  bool hasSYCL() const override;
  // bool hasMAGMA() const override;
  // bool hasSyCL() const override;
  // const at::sycl::NVRTC& nvrtc() const override;
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
  // int64_t syclFFTGetPlanCacheMaxSize(int64_t device_index) const override;
  // void syclFFTSetPlanCacheMaxSize(int64_t device_index, int64_t max_size) const override;
  // int64_t syclFFTGetPlanCacheSize(int64_t device_index) const override;
  // void syclFFTClearPlanCache(int64_t device_index) const override;
  int getNumGPUs() const override;
};

}}} // at::sycl::detail
