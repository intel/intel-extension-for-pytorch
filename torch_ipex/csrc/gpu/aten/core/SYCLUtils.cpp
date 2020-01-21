#include <stdlib.h>

#include <c10/macros/Macros.h>
#include <c10/core/Device.h>
#include <core/SYCLUtils.h>
#include <core/SYCLDevice.h>
#include <core/SYCLException.h>
#include <core/SYCLStream.h>
#include <core/SYCLContext.h>
#include <cmath>

namespace c10 {
namespace sycl {

// Global device pool state
static std::once_flag init_device_flag;
static SYCLDevicePool gDevPool;

static void clearSyclContextAndDevices() {
  at::sycl::clearGlobalContext();
  gDevPool.dev_sels.clear();
  gDevPool.devices.clear();
}

// It should be call only once. (std::call_once)
static void initGlobalDevicePoolState() {
  auto plaform_list = cl::sycl::platform::get_platforms();
  DeviceIndex devIndex = 0;
  for (const auto& platform : plaform_list) {
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        gDevPool.devices.push_back(device);
        gDevPool.dev_sels.push_back({device});
      }
    }
  }
  TORCH_CHECK(gDevPool.devices.size() > 0, "SYCL Device count is zero");
  gDevPool.cur_dev_index = 0;

  // Note: SYCLRuntime's destruction happens before the destroy of the
  // global vars except the global vars with sycl type. This will make
  // our global device pool destruction crash. So we use atexit to
  // manually free all sycl devices. atexit callback happens before
  // SYCLRuntime destruction.
  atexit(clearSyclContextAndDevices);
}

static void initDevicePoolCallOnce() {
  std::call_once(init_device_flag, initGlobalDevicePoolState);
}

int syclGetDeviceCount(int* deviceCount) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  *deviceCount = (int)gDevPool.devices.size();
  return SYCL_SUCCESS;
}

int syclGetDevice(DeviceIndex* pDI) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  TORCH_CHECK(pDI != NULL);
  *pDI = gDevPool.cur_dev_index;
  return SYCL_SUCCESS;
}

int syclSetDevice(DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (DeviceIndex)gDevPool.devices.size()) {
    AT_WARN("syclSetDevice: device_index is out of range");
  } else {
    gDevPool.cur_dev_index = device_index;
  }
  return SYCL_SUCCESS;
}

cl::sycl::device syclGetRawDevice(DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (DeviceIndex)gDevPool.devices.size()) {
    AT_ERROR("syclSetDevice: device_index is out of range");
  }
  return gDevPool.devices[device_index];
}

DPCPPDeviceSelector syclGetDeviceSelector(DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (DeviceIndex)gDevPool.devices.size()) {
    AT_ERROR("syclSetDevice: device_index is out of range");
  }
  return gDevPool.dev_sels[device_index];
}

/************************sycl memory buffer map pool****************/
#define MAX_SYCL_MEM_PER_DEVICE 17179869184 // 16*1024*1024*1024 -- 16GB
// Global buffer map pool state
static std::once_flag init_buffer_map_flag;
static std::vector<cl::sycl::codeplay::PointerMapper*> gBufferMapPoolPtr;
static void initBufferMapPoolStates() {
  int device_count;
  C10_SYCL_CHECK(syclGetDeviceCount(&device_count));
  gBufferMapPoolPtr.resize(device_count);
  for (int i = 0; i < device_count; i++) {
    gBufferMapPoolPtr[i] = new cl::sycl::codeplay::PointerMapper(
        4096 + i * MAX_SYCL_MEM_PER_DEVICE);
  }
}

static void initBufferMapPoolCallOnce() {
  std::call_once(init_buffer_map_flag, initBufferMapPoolStates);
}

cl::sycl::codeplay::PointerMapper& syclGetBufferMap() {
  initBufferMapPoolCallOnce();
  DeviceIndex device_id;
  C10_SYCL_CHECK(syclGetDevice(&device_id));
  return *gBufferMapPoolPtr[device_id];
}

int syclGetDeviceIdFromPtr(DeviceIndex* device_id, void* ptr) {
  int device_index = reinterpret_cast<uint64_t>(ptr) / MAX_SYCL_MEM_PER_DEVICE;
  int device_count;
  syclGetDeviceCount(&device_count);
  if (device_index >= device_count) {
    throw(std::out_of_range("this pointer is invalid"));
  }

  if (gBufferMapPoolPtr[device_index]->get_offset(ptr) > 0) {
    *device_id = static_cast<DeviceIndex>(device_index);
  } else {
    throw(std::out_of_range("the pointer is not allocated"));
  }
  return SYCL_SUCCESS;
}

cl::sycl::queue& syclGetCurrentQueue() {
  return getCurrentSYCLStream().sycl_queue();
}

int64_t syclMaxWorkGroupSize(cl::sycl::queue& queue) {
  return queue.get_device().get_info<dp_dev_max_wgroup_size>();
}

int64_t syclMaxWorkGroupSize() {
  auto& queue = syclGetCurrentQueue();
  return syclMaxWorkGroupSize(queue);
}

int64_t syclMaxComputeUnitSize(cl::sycl::queue& queue) {
  return queue.get_device().template get_info<cl::sycl::info::device::max_compute_units>();
}

int64_t syclMaxComputeUnitSize() {
  auto& queue = syclGetCurrentQueue();
  return syclMaxComputeUnitSize(queue);
}

void parallel_for_setup(int64_t n, int64_t& tileSize, int64_t& rng, int64_t& GRange) {
  tileSize = syclMaxWorkGroupSize();
  rng = n;
  if (rng == 0) {
    rng = static_cast<int64_t>(1);
  }

  GRange = rng;
  if (tileSize > GRange) {
    tileSize = GRange;
  } else if (GRange > tileSize) {
    int64_t xMode = static_cast<int64_t>(GRange % tileSize);
    if (xMode != 0) {
      GRange += static_cast<int64_t>(tileSize - xMode);
    }
  }
}

void parallel_for_setup(int64_t dim0, int64_t dim1,
                        int64_t& tileSize0, int64_t& tileSize1,
                        int64_t& rng0, int64_t& rng1,
                        int64_t& GRange0, int64_t& GRange1) {
  int64_t max_workgroup_Size = syclMaxWorkGroupSize();
  int64_t pow_of_2 = static_cast<int64_t>(std::log2(max_workgroup_Size));
  tileSize1 =
      static_cast<int64_t>(std::pow(2, static_cast<int64_t>(pow_of_2 / 2)));
  rng1 = dim1;
  if (rng1 == 0) {
    rng1 = static_cast<int64_t>(1);
  }

  GRange1 = rng1;
  if (tileSize1 > GRange1) {
    tileSize1 = GRange1;
  } else if (GRange1 > tileSize1) {
    int64_t xMode = static_cast<int64_t>(GRange1 % tileSize1);
    if (xMode != 0) {
      GRange1 += static_cast<int64_t>(tileSize1 - xMode);
    }
  }

  tileSize0 = static_cast<int64_t>(max_workgroup_Size / tileSize1);
  rng0 = dim0;
  if (rng0 == 0) {
    rng0 = static_cast<int64_t>(1);
  }

  GRange0 = rng0;
  if (tileSize0 > GRange0) {
    tileSize0 = GRange0;
  } else if (GRange0 > tileSize0) {
    int64_t xMode = static_cast<int64_t>(GRange0 % tileSize0);
    if (xMode != 0) {
      GRange0 += static_cast<int64_t>(tileSize0 - xMode);
    }
  }
}

void parallel_for_setup(int64_t dim0, int64_t dim1, int64_t dim2,
                        int64_t& tileSize0, int64_t& tileSize1, int64_t& tileSize2,
                        int64_t& rng0, int64_t& rng1, int64_t& rng2,
                        int64_t& GRange0, int64_t& GRange1, int64_t& GRange2) {
  int64_t max_workgroup_Size = syclMaxWorkGroupSize();
  int64_t pow_of_2 = static_cast<int64_t>(std::log2(max_workgroup_Size));
  tileSize2 =
      static_cast<int64_t>(std::pow(2, static_cast<int64_t>(pow_of_2 / 3)));
  rng2 = dim2;
  if (rng2 == 0) {
    rng1 = static_cast<int64_t>(1);
  }

  GRange2 = rng2;
  if (tileSize2 > GRange2) {
    tileSize2 = GRange2;
  } else if (GRange2 > tileSize2) {
    int64_t xMode = static_cast<int64_t>(GRange2 % tileSize2);
    if (xMode != 0)
      GRange2 += static_cast<int64_t>(tileSize2 - xMode);
  }

  pow_of_2 = static_cast<int64_t>(
      std::log2(static_cast<int64_t>(max_workgroup_Size / tileSize2)));
  tileSize1 = static_cast<int64_t>(std::pow(2, static_cast<int64_t>(pow_of_2 / 2)));

  rng1 = dim1;
  if (rng1 == 0) {
    rng1 = static_cast<int64_t>(1);
  }

  GRange1 = rng1;
  if (tileSize1 > GRange1) {
    tileSize1 = GRange1;
  } else if (GRange1 > tileSize1) {
    int64_t xMode = static_cast<int64_t>(GRange1 % tileSize1);
    if (xMode != 0) {
      GRange1 += static_cast<int64_t>(tileSize1 - xMode);
    }
  }

  tileSize0 = static_cast<int64_t>(max_workgroup_Size / (tileSize1 * tileSize2));
  rng0 = dim0;
  if (rng0 == 0) {
    rng0 = static_cast<int64_t>(1);
  }

  GRange0 = rng0;
  if (tileSize0 > GRange0) {
    tileSize0 = GRange0;
  } else if (GRange0 > tileSize0) {
    int64_t xMode = static_cast<int64_t>(GRange0 % tileSize0);
    if (xMode != 0) {
      GRange0 += static_cast<int64_t>(tileSize0 - xMode);
    }
  }
}

} // namespace sycl
} // namespace c10
