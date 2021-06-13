#include <runtime/Device.h>
#include <core/Device.h>
#include <core/Stream.h>
#include <runtime/Exception.h>
#include <runtime/Env.h>
#include <runtime/Context.h>

#include <cmath>
#include <mutex>
#include <vector>

namespace xpu {
namespace dpcpp {

// Global device pool state
static std::once_flag init_device_flag;
static std::once_flag init_prop_flag;
static std::deque<std::once_flag> device_prop_flags;
static std::vector<XPUDeviceProp> device_properties;
static thread_local at::DeviceIndex cur_dev_index = 0;

struct DPCPPDevicePool {
  std::vector<DPCPP::device> devices;
  std::mutex devices_mutex;
} gDevPool;

static void clearDPCPPContextAndDevices() {
  xpu::dpcpp::clearDeviceContext();
  gDevPool.devices.clear();
}

// It should be call only once. (std::call_once)
static void initGlobalDevicePoolState() {
  auto plaform_list = DPCPP::platform::get_platforms();
  std::vector<DPCPP::device> root_devices;
  // Enumerated root devices(GPU cards) from GPU Platform firstly.
  for (const auto& platform : plaform_list) {
#ifdef USE_LEVEL_ZERO_ONLY
    if (platform.get_backend() != DPCPP::backend::level_zero)
      continue;
#else
    auto plat_name = platform.get_info<DPCPP::info::platform::name>();
    if (plat_name.compare(getPreferredPlatform()) != 0)
      continue;
#endif
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }

  // Mapping framework device to physical tile by default.
  // If IPEX_DISABLE_TILE_PARTITION enabled, mapping framework device to physical device.
  if (disable_tile_partition()) {
    gDevPool.devices = std::move(root_devices);
  } else {
    constexpr DPCPP::info::partition_property partition_by_affinity =
      DPCPP::info::partition_property::partition_by_affinity_domain;
    constexpr DPCPP::info::partition_affinity_domain next_partitionable =
      DPCPP::info::partition_affinity_domain::next_partitionable;
    for (const auto &root_device : root_devices) {
      std::vector<DPCPP::device> sub_devices;
      try {
        sub_devices = root_device.create_sub_devices<partition_by_affinity>(next_partitionable);
        gDevPool.devices.insert(gDevPool.devices.end(), sub_devices.begin(), sub_devices.end());
      } catch (DPCPP::feature_not_supported &e) {
        TORCH_WARN("Tile partition is not supported on this device: ", root_device.get_info<dpcpp_dev_name>());
        gDevPool.devices.push_back(root_device);
      }
    }
  }

  auto device_count = gDevPool.devices.size();
  TORCH_CHECK(device_count > 0, "DPCPP Device count is zero");

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

int dpcppGetDevice(at::DeviceIndex* pDI) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  TORCH_CHECK(pDI != NULL);
  *pDI = cur_dev_index;
  return DPCPP_SUCCESS;
}

int dpcppSetDevice(at::DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (at::DeviceIndex)gDevPool.devices.size()) {
    TORCH_WARN("dpcppSetDevice: device_index is out of range");
  } else {
    cur_dev_index = device_index;
  }
  return DPCPP_SUCCESS;
}

DPCPP::device dpcppGetRawDevice(at::DeviceIndex device_index) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_index >= (at::DeviceIndex)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppSetDevice: device_index is out of range");
  }
  return gDevPool.devices[device_index];
}

at::DeviceIndex dpcppGetDeviceIndex(DPCPP::device device) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  auto it = std::find(gDevPool.devices.begin(), gDevPool.devices.end(), device);
  if (it != gDevPool.devices.end()) {
    return std::distance(gDevPool.devices.begin(), it);
  }
  return -1;
}

int dpcppGetDeviceIdFromPtr(at::DeviceIndex* device_id, void* ptr) {
  auto raw_device = DPCPP::get_pointer_device(ptr, xpu::dpcpp::getDeviceContext());
  *device_id = dpcppGetDeviceIndex(raw_device);
  return DPCPP_SUCCESS;
}

static inline std::string getPreferredPlatform() {
  // TODO: To use more stable api from dpc++ runtime to preferred select
  // platform Following code logic based upon the assumption: gpu_selector will
  // select gpu device with priority considering platform: 1) level_zero 2)
  // opencl JIRA CMPLRLLVM-19937 is tracking this.
  DPCPP::device dev{DPCPP::gpu_selector{}};
  return dev.get_platform().get_info<DPCPP::info::platform::name>();
}

XPUDeviceProp* getCurrentDeviceProperties() {
  auto device = current_device();
  return getDeviceProperties(device);
}

void initXPUContextVectors() {
  auto num_gpus = device_count();
  device_prop_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  XPUDeviceProp device_prop;
  auto device = dpcppGetRawDevice(device_index);
  device_prop.name = device.get_info<dpcpp_dev_name>();
  device_prop.dev_type = device.get_info<dpcpp_dev_type>();
  device_prop.total_global_mem = device.get_info<dpcpp_dev_global_mem_size>();
  device_prop.max_compute_units = device.get_info<dpcpp_dev_max_units>();
  device_prop.platform_name = device.get_info<DPCPP::info::device::platform>().get_info<DPCPP::info::platform::name>();
  device_prop.sub_devices_number = device.get_info<DPCPP::info::device::partition_max_sub_devices>();
  device_properties[device_index] = device_prop;
}

XPUDeviceProp* getDeviceProperties(int64_t device) {
  std::call_once(init_prop_flag, initXPUContextVectors);
  if (device == -1) device = current_device();
  auto num_gpus = device_count();
  AT_ASSERT(device >= 0 && device < num_gpus);
  std::call_once(device_prop_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

} // namespace dpcpp
} // namespace at
