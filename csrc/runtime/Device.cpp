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
  std::vector<std::unique_ptr<sycl::device>> devices;
  // Record all device ids for each card
  std::vector<std::vector<int>> deviceids_card;
#if defined(USE_MULTI_CONTEXT)
  std::vector<std::unique_ptr<sycl::context>> contexts;
#endif
  std::mutex devices_mutex;
} gDevPool;

static void enumDevices(
    std::vector<std::unique_ptr<sycl::device>>& devices,
    std::vector<std::vector<int>>& deviceids_card) {
  std::vector<sycl::device> root_devices;
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated root devices(GPU cards) from GPU Platform firstly.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero)
      continue;
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }

  // Mapping framework device to physical tile by default.
  // If IPEX_TILE_AS_DEVICE disabled, mapping framework device to
  // physical device.
  if (Settings::I().is_tile_as_device_enabled()) {
    constexpr sycl::info::partition_property partition_by_affinity =
        sycl::info::partition_property::partition_by_affinity_domain;
    constexpr sycl::info::partition_affinity_domain next_partitionable =
        sycl::info::partition_affinity_domain::next_partitionable;
    for (const auto& root_device : root_devices) {
      std::vector<int> deviceids_eachcard = {};
      try {
        auto sub_devices =
            root_device.create_sub_devices<partition_by_affinity>(
                next_partitionable);
        for (auto& s_dev : sub_devices) {
          devices.push_back(std::make_unique<sycl::device>(s_dev));
          deviceids_eachcard.push_back(devices.size() - 1);
        }
        deviceids_card.push_back(deviceids_eachcard);
      } catch (sycl::exception& e) {
        // FIXME: should only check feature_not_supported here.
        // But for now we got invalid here if partition is not supported.
        if (e.code() != sycl::errc::feature_not_supported &&
            e.code() != sycl::errc::invalid) {
          throw std::runtime_error(
              std::string("Failed to apply tile partition: ") + e.what());
        }
        static auto verbose = Settings::I().get_verbose_level();
        if (verbose) {
          TORCH_WARN(
              "Tile partition is UNSUPPORTED : ",
              root_device.get_info<dpcpp_dev_name>());
        }
        devices.push_back(std::make_unique<sycl::device>(root_device));
      }
    }
  } else {
    for (const auto& root_device : root_devices) {
      std::vector<int> deviceids_eachcard = {};
      devices.push_back(std::make_unique<sycl::device>(root_device));
      deviceids_eachcard.push_back(devices.size() - 1);
      deviceids_card.push_back(deviceids_eachcard);
    }
  }
}

// It should be call only once. (std::call_once)
static void initGlobalDevicePoolState() {
  enumDevices(gDevPool.devices, gDevPool.deviceids_card);

  auto device_count = gDevPool.devices.size();
  if (device_count <= 0) {
    TORCH_WARN("DPCPP Device count is zero!");
  }

#if defined(USE_MULTI_CONTEXT)
  gDevPool.contexts.resize(device_count);
  for (int i = 0; i < device_count; i++) {
    gDevPool.contexts[i] = std::make_unique<sycl::context>(
        sycl::context({*gDevPool.devices[i]}, dpcppAsyncHandler));
  }
#endif
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

sycl::device dpcppGetRawDevice(DeviceId device_id) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_id >= (DeviceId)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppSetDevice: device_id is out of range");
  }
  return *gDevPool.devices[device_id];
}

DeviceId dpcppGetDeviceIndex(sycl::device device) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  auto comp_op = [&](std::unique_ptr<sycl::device>& dev) -> bool {
    return device == *dev;
  };
  auto it =
      std::find_if(gDevPool.devices.begin(), gDevPool.devices.end(), comp_op);
  if (it != gDevPool.devices.end()) {
    return std::distance(gDevPool.devices.begin(), it);
  }
  return -1;
}

sycl::context dpcppGetDeviceContext(DeviceId device) {
  initDevicePoolCallOnce();
  DeviceId device_id = device;
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
#if defined(USE_MULTI_CONTEXT)
  return *gDevPool.contexts[device_id];
#else
  auto dev = dpcppGetRawDevice(device_id);
  return dev.get_platform().ext_oneapi_get_default_context();
#endif
}

int dpcppGetDeviceIdFromPtr(DeviceId* device_id, void* ptr) {
  auto raw_device = sycl::get_pointer_device(ptr, dpcppGetDeviceContext());
  *device_id = dpcppGetDeviceIndex(raw_device);
  return DPCPP_SUCCESS;
}

static void initDevPropVectors() {
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

  // intel extensions
  device_prop.gpu_eu_simd_width = device.has(dpcpp_dev_aspect_gpu_eu_simd_width)
      ? device.get_info<dpcpp_dev_ext_intel_gpu_eu_simd_width>()
      : 8;
  device_prop.gpu_hw_threads_per_eu =
      device.has(dpcpp_dev_aspect_hw_threads_per_eu)
      ? device.get_info<dpcpp_dev_ext_intel_gpu_hw_threads_per_eu>()
      : 8;
  device_prop.support_atomic64 = device.has(dpcpp_dev_aspect_atomic64);
  device_prop.support_fp64 = device.has(dpcpp_dev_aspect_fp64);

  device_properties[device_id] = device_prop;
}

DeviceProp* dpcppGetCurrentDeviceProperties() {
  DeviceId device = 0;
  AT_DPCPP_CHECK(dpcppGetDevice(&device));
  return dpcppGetDeviceProperties(device);
}

DeviceProp* dpcppGetDeviceProperties(DeviceId device) {
  std::call_once(init_prop_flag, initDevPropVectors);
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

// By default, get the first card id that contains max number of devices.
static int getMaxTilesCardId(std::vector<std::vector<int>>& deviceids_card) {
  int card_id = -1;
  int maxnum_devices = 0;
  int num_cards = deviceids_card.size();
  for (int i = 0; i < num_cards; i++) {
    auto num_devices = deviceids_card[i].size();
    if (maxnum_devices < num_devices) {
      maxnum_devices = num_devices;
      card_id = i;
    }
  }
  return card_id;
}

std::vector<int>& dpcppGetDeviceIdListForCard(int card_id) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (card_id == -1) {
    card_id = getMaxTilesCardId(gDevPool.deviceids_card);
  }
  TORCH_CHECK(
      card_id >= 0 && card_id < gDevPool.deviceids_card.size(),
      "card_id ",
      card_id,
      " out of range");
  return gDevPool.deviceids_card[card_id];
}

// This function can be used to prefetch device count and no execption. It is
// used in device_count() and is_available() such that both two functions can be
// called before forking process.
int dpcppPrefetchDeviceCount() noexcept {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::vector<std::vector<int>> deviceids_card;
  enumDevices(devices, deviceids_card);
  return devices.size();
}

// This function can be used to prefetch device list for each card. It is used
// in getDeviceIdListForCard() such that this function can be called before
// forking process.
std::vector<int> dpcppPrefetchDeviceIdListForCard(int card_id) {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::vector<std::vector<int>> deviceids_card;
  enumDevices(devices, deviceids_card);
  if (card_id == -1) {
    card_id = getMaxTilesCardId(deviceids_card);
  }
  TORCH_CHECK(
      card_id >= 0 && card_id < deviceids_card.size(),
      "card_id ",
      card_id,
      " out of range");
  return deviceids_card[card_id];
}

} // namespace dpcpp
} // namespace xpu
