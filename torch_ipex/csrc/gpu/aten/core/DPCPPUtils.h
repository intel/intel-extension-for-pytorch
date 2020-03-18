#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/Device.h>

#include <core/virtual_ptr.h>
#include <core/Device.h>
#include <core/DPCPP.h>


namespace at {
namespace dpcpp {

// move to c10/dpcpp/DPCPPFunctions.h
// DeviceIndex device_count();

// move to c10/dpcpp/DPCPPFunctions.h
// DeviceIndex current_device();

// move to c10/dpcpp/DPCPPFunctions.h
// void set_device(DeviceIndex device);

int dpcppGetDeviceCount(int *deviceCount);

int dpcppGetDevice(DeviceIndex *pDI);

int dpcppSetDevice(DeviceIndex device_index);

int dpcppGetDeviceIdFromPtr(DeviceIndex *device_id, void *ptr);

DP::device dpcppGetRawDevice(DeviceIndex device_index);

DPCPPDeviceSelector dpcppGetDeviceSelector(DeviceIndex device_index);

DP::codeplay::PointerMapper &dpcppGetBufferMap();

DP::queue &dpcppGetCurrentQueue();

int64_t dpcppMaxWorkGroupSize();

int64_t dpcppMaxWorkGroupSize(DP::queue &queue);

int64_t dpcppMaxComputeUnitSize();

int64_t dpcppMaxComputeUnitSize(DP::queue& queue);


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

#include <core/Exception.h>
static DP::async_handler dpcppAsyncHandler = [](DP::exception_list eL) {
  for (auto& e : eL) {
    AT_DPCPP_TRY
    std::rethrow_exception(e);
    AT_DPCPP_CATCH_RETHROW(__FILE__, __LINE__)
  }
};

}} // namespace at::dpcpp
