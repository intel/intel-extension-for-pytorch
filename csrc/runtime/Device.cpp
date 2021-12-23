#include <runtime/Context.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include <utils/Settings.h>

#include <cmath>
#include <deque>
#include <mutex>
#include <vector>

namespace xpu {
namespace dpcpp {

// Global device pool state
static std::once_flag init_device_flag;
static std::once_flag init_prop_flag;
static std::deque<std::once_flag> device_prop_flags;
static std::vector<DeviceProp> device_properties;
static thread_local DeviceId cur_dev_index = 0;

struct DPCPPDevicePool {
  std::vector<DPCPP::device> devices;
  std::mutex devices_mutex;
} gDevPool;

static void clearDPCPPContextAndDevices() {
  clearDeviceContext();
  gDevPool.devices.clear();
}

static inline std::string getPreferredPlatform() {
  // TODO: To use more stable api from dpc++ runtime to preferred select
  // platform Following code logic based upon the assumption: gpu_selector will
  // select gpu device with priority considering platform: 1) level_zero 2)
  // opencl JIRA CMPLRLLVM-19937 is tracking this.
  DPCPP::device dev{DPCPP::gpu_selector{}};
  return dev.get_platform().get_info<DPCPP::info::platform::name>();
}

// It should be call only once. (std::call_once)
static void initGlobalDevicePoolState() {
  auto plaform_list = DPCPP::platform::get_platforms();
  std::vector<DPCPP::device> root_devices;
  // Enumerated root devices(GPU cards) from GPU Platform firstly.
  for (const auto& platform : plaform_list) {
#ifdef USE_LEVEL_ZERO_ONLY
    if (platform.get_backend() != DPCPP::backend::ext_oneapi_level_zero)
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
  // If IPEX_DISABLE_TILE_PARTITION enabled, mapping framework device to
  // physical device.
  if (Settings::I().is_tile_partition_enabled()) {
    constexpr DPCPP::info::partition_property partition_by_affinity =
        DPCPP::info::partition_property::partition_by_affinity_domain;
    constexpr DPCPP::info::partition_affinity_domain next_partitionable =
        DPCPP::info::partition_affinity_domain::next_partitionable;
    for (const auto& root_device : root_devices) {
      std::vector<DPCPP::device> sub_devices;
      try {
        sub_devices = root_device.create_sub_devices<partition_by_affinity>(
            next_partitionable);
        gDevPool.devices.insert(
            gDevPool.devices.end(), sub_devices.begin(), sub_devices.end());
      } catch (sycl::exception& e) {
        // FIXME: should only check feature_not_supported here.
        // But for now we got invalid here if partition is not supported.
        if (e.code() != DPCPP::errc::feature_not_supported &&
            e.code() != DPCPP::errc::invalid) {
          throw std::runtime_error(
              std::string("Failed to apply tile partition: ") + e.what());
        }
        TORCH_WARN(
            "Tile partition is UNSUPPORTED : ",
            root_device.get_info<dpcpp_dev_name>());
        gDevPool.devices.push_back(root_device);
      }
    }
  } else {
    gDevPool.devices = std::move(root_devices);
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

int dpcppGetDevice(DeviceId* pDI) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  TORCH_CHECK(pDI != NULL);
  *pDI = cur_dev_index;
  return DPCPP_SUCCESS;
}

int dpcppSetDevice(DeviceId device_id) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_id >= (DeviceId)gDevPool.devices.size()) {
    TORCH_WARN("dpcppSetDevice: device_id is out of range");
  } else {
    cur_dev_index = device_id;
  }
  return DPCPP_SUCCESS;
}

DPCPP::device dpcppGetRawDevice(DeviceId device_id) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_id >= (DeviceId)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppSetDevice: device_id is out of range");
  }
  return gDevPool.devices[device_id];
}

DeviceId dpcppGetDeviceIndex(DPCPP::device device) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  auto it = std::find(gDevPool.devices.begin(), gDevPool.devices.end(), device);
  if (it != gDevPool.devices.end()) {
    return std::distance(gDevPool.devices.begin(), it);
  }
  return -1;
}

int dpcppGetDeviceIdFromPtr(DeviceId* device_id, void* ptr) {
  auto raw_device = DPCPP::get_pointer_device(ptr, getDeviceContext());
  *device_id = dpcppGetDeviceIndex(raw_device);
  return DPCPP_SUCCESS;
}

static void initContextVectors() {
  auto num_gpus = 0;
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&num_gpus));
  device_prop_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

static void initDeviceProperty(DeviceId device_id) {
  DeviceProp device_prop;
  auto device = dpcppGetRawDevice(device_id);

  device_prop.dev_name = device.get_info<dpcpp_dev_name>();
  device_prop.dev_type = device.get_info<dpcpp_dev_type>();
  device_prop.platform_name =
      device.get_info<dpcpp_dev_platform>().get_info<dpcpp_platform_name>();
  device_prop.vendor = device.get_info<dpcpp_dev_vendor>();
  device_prop.driver_version = device.get_info<dpcpp_dev_driver_version>();
  device_prop.version = device.get_info<dpcpp_dev_version>();
  // device_prop.backend_version = device.get_info<dpcpp_dev_backend_version>();
  device_prop.is_available = device.get_info<dpcpp_dev_is_available>();
  device_prop.max_param_size = device.get_info<dpcpp_dev_max_param_size>();
  device_prop.max_compute_units =
      device.get_info<dpcpp_dev_max_compute_units>();
  device_prop.max_work_item_dims =
      device.get_info<dpcpp_dev_max_work_item_dims>();
  device_prop.max_work_group_size =
      device.get_info<dpcpp_dev_max_work_group_size>();
  device_prop.max_num_subgroup = device.get_info<dpcpp_dev_max_num_subgroup>();
  device_prop.subgroup_sizes = device.get_info<dpcpp_dev_subgroup_sizes>();
  device_prop.max_clock_freq = device.get_info<dpcpp_dev_max_clock_freq>();
  device_prop.address_bits = device.get_info<dpcpp_dev_address_bits>();
  device_prop.max_mem_alloc_size = device.get_info<dpcpp_dev_max_alloc_size>();
  device_prop.base_addr_align =
      device.get_info<dpcpp_dev_mem_base_addr_align>();
  device_prop.half_fp_config = device.get_info<dpcpp_dev_half_fp_config>();
  device_prop.single_fp_config = device.get_info<dpcpp_dev_single_fp_config>();
  device_prop.double_fp_config = device.get_info<dpcpp_dev_double_fp_config>();
  device_prop.global_mem_size = device.get_info<dpcpp_dev_global_mem_size>();
  device_prop.global_mem_cache_type =
      device.get_info<dpcpp_dev_global_mem_cache_type>();
  device_prop.global_mem_cache_size =
      device.get_info<dpcpp_dev_global_mem_cache_size>();
  device_prop.global_mem_cache_line_size =
      device.get_info<dpcpp_dev_global_mem_cache_line_size>();
  device_prop.local_mem_type = device.get_info<dpcpp_dev_local_mem_type>();
  device_prop.local_mem_size = device.get_info<dpcpp_dev_local_mem_size>();
  device_prop.max_sub_devices = device.get_info<dpcpp_dev_max_sub_devices>();
  device_prop.profiling_resolution =
      device.get_info<dpcpp_dev_profiling_resolution>();

  device_prop.pref_vec_width_char =
      device.get_info<dpcpp_dev_pref_vec_width_char>();
  device_prop.pref_vec_width_short =
      device.get_info<dpcpp_dev_pref_vec_width_short>();
  device_prop.pref_vec_width_int =
      device.get_info<dpcpp_dev_pref_vec_width_int>();
  device_prop.pref_vec_width_long =
      device.get_info<dpcpp_dev_pref_vec_width_long>();
  device_prop.pref_vec_width_float =
      device.get_info<dpcpp_dev_pref_vec_width_float>();
  device_prop.pref_vec_width_double =
      device.get_info<dpcpp_dev_pref_vec_width_double>();
  device_prop.pref_vec_width_half =
      device.get_info<dpcpp_dev_pref_vec_width_half>();

  device_prop.native_vec_width_char =
      device.get_info<dpcpp_dev_native_vec_width_char>();
  device_prop.native_vec_width_short =
      device.get_info<dpcpp_dev_native_vec_width_short>();
  device_prop.native_vec_width_int =
      device.get_info<dpcpp_dev_native_vec_width_int>();
  device_prop.native_vec_width_long =
      device.get_info<dpcpp_dev_native_vec_width_long>();
  device_prop.native_vec_width_float =
      device.get_info<dpcpp_dev_native_vec_width_float>();
  device_prop.native_vec_width_double =
      device.get_info<dpcpp_dev_native_vec_width_double>();
  device_prop.native_vec_width_half =
      device.get_info<dpcpp_dev_native_vec_width_half>();

  device_properties[device_id] = device_prop;
}

DeviceProp* dpcppGetCurrentDeviceProperties() {
  DeviceId device = 0;
  AT_DPCPP_CHECK(dpcppGetDevice(&device));
  return dpcppGetDeviceProperties(device);
}

DeviceProp* dpcppGetDeviceProperties(DeviceId device) {
  std::call_once(init_prop_flag, initContextVectors);
  DeviceId device_id = device;
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  auto num_gpus = 0;
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&num_gpus));
  AT_ASSERT(device_id >= 0 && device_id < num_gpus);
  std::call_once(device_prop_flags[device_id], initDeviceProperty, device_id);
  return &device_properties[device_id];
}

} // namespace dpcpp
} // namespace xpu
