#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <core/DPCPP.h>
#include <core/Device.h>


using namespace at;

namespace xpu {
namespace dpcpp {

int dpcppGetDeviceCount(int* deviceCount);

int dpcppGetDevice(DeviceIndex* pDI);

int dpcppSetDevice(DeviceIndex device_index);

int dpcppGetDeviceIdFromPtr(DeviceIndex* device_id, void* ptr);

DPCPP::device dpcppGetRawDevice(DeviceIndex device_index);

DPCPPDeviceSelector dpcppGetDeviceSelector(DeviceIndex device_index);

DPCPP::queue& dpcppGetCurrentQueue();

int64_t dpcppMaxWorkGroupSize();

int64_t dpcppMaxWorkGroupSize(DPCPP::queue& queue);

int64_t dpcppMaxComputeUnitSize();

int64_t dpcppMaxComputeUnitSize(DPCPP::queue& queue);

int64_t dpcppMaxDSSNum();

int64_t dpcppMaxDSSNum(DPCPP::queue& queue);

int64_t dpcppLocalMemSize();

int64_t dpcppLocalMemSize(DPCPP::queue& queue);

std::string getPreferredPlatform();

void parallel_for_setup(
    int64_t n,
    int64_t& tileSize,
    int64_t& rng,
    int64_t& GRange);

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& GRange0,
    int64_t& GRange1);

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& tileSize2,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& rng2,
    int64_t& GRange0,
    int64_t& GRange1,
    int64_t& GRange2);

union u32_to_f32 {
  uint32_t in;
  float out;
};

union f32_to_u32 {
  float in;
  uint32_t out;
};

union double_to_ull{
  double in;
  unsigned long long out;
};

union ull_to_double{
  unsigned long long in;
  double out;
};

union double_to_u32{
  double in;
  struct { uint32_t high, low;};
};

static inline DPCPP_DEVICE uint32_t __float_as_int(float val) {
  f32_to_u32 cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE float __int_as_float(uint32_t val) {
  u32_to_f32 cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE uint32_t __double_as_long_long(double val) {
  double_to_ull cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE float __long_long_as_double(unsigned long long val) {
  ull_to_double cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE uint32_t __double_as_int(double val) {
  double_to_u32 cn;
  cn.in = val;
  return cn.low;
}

static inline DPCPP_DEVICE float __int_as_double(uint32_t val) {
  return (double)val;
}

#include <core/Exception.h>
static DPCPP::async_handler dpcppAsyncHandler = [](DPCPP::exception_list eL) {
  for (auto& e : eL) {
    AT_DPCPP_TRY
    std::rethrow_exception(e);
    AT_DPCPP_CATCH_RETHROW(__FILE__, __LINE__)
  }
};

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define THREAD_WORK_SIZE 4
#define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * num_threads)
#define thread_work_size THREAD_WORK_SIZE

} // namespace dpcpp
} // namespace xpu
