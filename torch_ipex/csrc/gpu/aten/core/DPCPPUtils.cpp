#include <stdlib.h>

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Device.h>
#include <core/Exception.h>
#include <core/Stream.h>
#include <cmath>

namespace at {
namespace dpcpp {

// Global device pool state
static std::once_flag init_device_flag;
static DPCPPDevicePool gDevPool;

static void clearDPCPPContextAndDevices() {
  at::dpcpp::clearGlobalContext();
  gDevPool.dev_sels.clear();
  gDevPool.devices.clear();
}

// It should be call only once. (std::call_once)
static void initGlobalDevicePoolState() {
  auto plaform_list = DPCPP::platform::get_platforms();
  DeviceIndex devIndex = 0;
  for (const auto& platform : plaform_list) {
    auto plat_name = platform.get_info<DPCPP::info::platform::name>();
    if (plat_name.compare(getPreferredPlatform()) != 0)
      continue;
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        gDevPool.devices.push_back(device);
        gDevPool.dev_sels.push_back({device});
      }
    }
  }
  TORCH_CHECK(gDevPool.devices.size() > 0, "DPCPP Device count is zero");
  gDevPool.cur_dev_index = 0;

  // Note: DPCPPRuntime's destruction happens before the destroy of the
  // global vars except the global vars with dpcpp type. This will make
  // our global device pool destruction crash. So we use atexit to
  // manually free all dpcpp devices. atexit callback happens before
  // DPCPPRuntime destruction.
  atexit(clearDPCPPContextAndDevices);
}

static void initDevicePoolCallOnce() {
  std::call_once(init_device_flag, initGlobalDevicePoolState);
}

int dpcppGetDeviceCount(int* deviceCount) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  *deviceCount = (int)gDevPool.devices.size();
  return DPCPP_SUCCESS;
}

int dpcppGetDevice(DeviceIndex* pDI) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  TORCH_CHECK(pDI != NULL);
  *pDI = gDevPool.cur_dev_index;
  return DPCPP_SUCCESS;
}

int dpcppSetDevice(DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (DeviceIndex)gDevPool.devices.size()) {
    TORCH_WARN("dpcppSetDevice: device_index is out of range");
  } else {
    gDevPool.cur_dev_index = device_index;
  }
  return DPCPP_SUCCESS;
}

DPCPP::device dpcppGetRawDevice(DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (DeviceIndex)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppSetDevice: device_index is out of range");
  }
  return gDevPool.devices[device_index];
}

DPCPPDeviceSelector dpcppGetDeviceSelector(DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (DeviceIndex)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppSetDevice: device_index is out of range");
  }
  return gDevPool.dev_sels[device_index];
}

DeviceIndex dpcppGetDeviceIndex(DPCPP::device device) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  auto it = std::find(gDevPool.devices.begin(), gDevPool.devices.end(), device);
  if (it != gDevPool.devices.end()) {
    return std::distance(gDevPool.devices.begin(), it);
  }
  return -1;
}

/************************dpcpp memory buffer map pool****************/
#define MAX_DPCPP_MEM_PER_DEVICE 17179869184 // 16*1024*1024*1024 -- 16GB
// Global buffer map pool state
static std::once_flag init_buffer_map_flag;
static std::vector<DPCPP::codeplay::PointerMapper*> gBufferMapPoolPtr;
static void initBufferMapPoolStates() {
  int device_count;
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&device_count));
  gBufferMapPoolPtr.resize(device_count);
  for (int i = 0; i < device_count; i++) {
    gBufferMapPoolPtr[i] = new DPCPP::codeplay::PointerMapper(
        4096 + i * MAX_DPCPP_MEM_PER_DEVICE);
  }
}

static void initBufferMapPoolCallOnce() {
  std::call_once(init_buffer_map_flag, initBufferMapPoolStates);
}

DPCPP::codeplay::PointerMapper& dpcppGetBufferMap() {
#ifndef USE_USM
  initBufferMapPoolCallOnce();
  DeviceIndex device_id;
  AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  return *gBufferMapPoolPtr[device_id];
#else
  throw(std::runtime_error("Invalid call get sycl buffer map in USM mode"));
#endif
}

int dpcppGetDeviceIdFromPtr(DeviceIndex* device_id, void* ptr) {
#ifndef USE_USM
  int device_index = reinterpret_cast<uint64_t>(ptr) / MAX_DPCPP_MEM_PER_DEVICE;
  int device_count;
  dpcppGetDeviceCount(&device_count);
  if (device_index >= device_count) {
    throw(std::out_of_range("this pointer is invalid"));
  }
  if (gBufferMapPoolPtr[device_index]->get_offset(ptr) > 0) {
    *device_id = static_cast<DeviceIndex>(device_index);
  } else {
    throw(std::out_of_range("the pointer is not allocated"));
  }
#else
  auto raw_device = DPCPP::get_pointer_device(ptr, at::dpcpp::getGlobalContext());
  *device_id = dpcppGetDeviceIndex(raw_device);
#endif
  return DPCPP_SUCCESS;
}

DPCPP::queue& dpcppGetCurrentQueue() {
  return getCurrentDPCPPStream().dpcpp_queue();
}

int64_t dpcppMaxWorkGroupSize(DPCPP::queue& queue) {
  return queue.get_device().get_info<dpcpp_dev_max_wgroup_size>();
}

int64_t dpcppMaxWorkGroupSize() {
  auto& queue = dpcppGetCurrentQueue();
  return dpcppMaxWorkGroupSize(queue);
}

int64_t dpcppMaxComputeUnitSize(DPCPP::queue& queue) {
  return queue.get_device()
      .template get_info<DPCPP::info::device::max_compute_units>();
}

int64_t dpcppMaxComputeUnitSize() {
  auto& queue = dpcppGetCurrentQueue();
  return dpcppMaxComputeUnitSize(queue);
}

std::string getPreferredPlatform() {
  // TODO: To use more stable api from dpc++ runtime to preferred select
  // platform Following code logic based upon the assumption: gpu_selector will
  // select gpu device with priority considering platform: 1) level_zero 2)
  // opencl JIRA CMPLRLLVM-19937 is tracking this.
  DPCPP::device dev{DPCPP::gpu_selector{}};
  return dev.get_platform().get_info<DPCPP::info::platform::name>();
}

void parallel_for_setup(
    int64_t n,
    int64_t& tileSize,
    int64_t& rng,
    int64_t& GRange) {
  tileSize = dpcppMaxWorkGroupSize();
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

void parallel_for_setup(
    int64_t dim0,
    int64_t dim1,
    int64_t& tileSize0,
    int64_t& tileSize1,
    int64_t& rng0,
    int64_t& rng1,
    int64_t& GRange0,
    int64_t& GRange1) {
  int64_t max_workgroup_Size = dpcppMaxWorkGroupSize();
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
    int64_t& GRange2) {
  int64_t max_workgroup_Size = dpcppMaxWorkGroupSize();
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

  tileSize0 =
      static_cast<int64_t>(max_workgroup_Size / (tileSize1 * tileSize2));
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

} // namespace dpcpp
} // namespace at
