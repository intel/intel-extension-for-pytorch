#pragma once

// This header provides C++ wrappers around commonly used SYCL API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.dpcpp

#include <c10/macros/Macros.h>
#include <c10/core/Device.h>

#include <core/virtual_ptr.h>
#include <core/SYCLDevice.h>
#include <core/SYCL.h>

namespace c10 {
namespace sycl {

// move to c10/dpcpp/SYCLFunctions.h
// DeviceIndex device_count();

// move to c10/dpcpp/SYCLFunctions.h
// DeviceIndex current_device();

// move to c10/dpcpp/SYCLFunctions.h
// void set_device(DeviceIndex device);

int syclGetDeviceCount(int *deviceCount);

int syclGetDevice(DeviceIndex *pDI);

int syclSetDevice(DeviceIndex device_index);

int syclGetDeviceIdFromPtr(DeviceIndex *device_id, void *ptr);

cl::sycl::device syclGetRawDevice(DeviceIndex device_index);

DPCPPDeviceSelector syclGetDeviceSelector(DeviceIndex device_index);

cl::sycl::codeplay::PointerMapper &syclGetBufferMap();

cl::sycl::queue &syclGetCurrentQueue();

int64_t syclMaxWorkGroupSize();

int64_t syclMaxWorkGroupSize(cl::sycl::queue &queue);

int64_t syclMaxComputeUnitSize();

int64_t syclMaxComputeUnitSize(cl::sycl::queue& queue);


void parallel_for_setup(int64_t n, int64_t &tileSize, int64_t &rng, int64_t &GRange);

void parallel_for_setup(int64_t dim0, int64_t dim1,
                        int64_t &tileSize0, int64_t &tileSize1,
                        int64_t &rng0, int64_t &rng1,
                        int64_t &GRange0, int64_t &GRange1);

void parallel_for_setup(int64_t dim0, int64_t dim1,int64_t dim2,
                        int64_t &tileSize0, int64_t &tileSize1, int64_t &tileSize2,
                        int64_t &rng0, int64_t &rng1, int64_t &rng2,
                        int64_t &GRange0, int64_t &GRange1, int64_t &GRange2);

union u32_to_f32{
  uint32_t in;
  float out;
};

union f32_to_u32{
  float in;
  uint32_t out;
};

static inline DP_DEVICE uint32_t __float_as_int(float val){
  f32_to_u32 cn;
  cn.in = val;
  return cn.out;
}

static inline DP_DEVICE float __int_as_float(uint32_t val) {
  u32_to_f32 cn;
  cn.in = val;
  return cn.out;
}

#include <core/SYCLException.h>
static cl::sycl::async_handler syclAsyncHandler = [](cl::sycl::exception_list eL) {
  for (auto& e : eL) {
    C10_SYCL_TRY
    std::rethrow_exception(e);
    C10_SYCL_CATCH_RETHROW(__FILE__, __LINE__)
  }
};

}} // namespace c10::sycl
