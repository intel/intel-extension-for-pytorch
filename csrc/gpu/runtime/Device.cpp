#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include <utils/Settings.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
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
static std::vector<DeviceInfo> device_info;
static std::vector<DeviceProp> device_properties;
static thread_local DeviceId cur_dev_index = 0;

/*
 * Device hierarchy note.
 *
 * `ZE_FLAT_DEVICE_HIERARCHY`, a driver environment variable, allows users to
 * select the device hierarchy model with which the underlying hardware is
 * exposed and the types of devices returned with SYCL runtime.
 *
 * When setting to `COMPOSITE`, all root-devices are returned and traversing the
 * device hierarchy is possible, each containing sub-devices and implicit
 * scaling support.
 *
 * When setting to `FLAT`, all sub-devices are returned and traversing the
 * device hierarchy is NOT possible. So we can NOT access the their root
 * devices.
 *
 * When setting to `COMBINED`, it combined `COMPOSITE` and `FLAT` mode. All
 * sub-devices are returned and traversing the device hierarchy is possible. By
 * default, driver selects `FLAT` mode, where all sub-devices are exposed.
 *
 */

struct DPCPPDevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  // If macro USE_MULTI_CONTEXT is enabled, contexts will be constructed by SYCL
  // runtime API sycl::context. Otherwise, contexts will be initialized by
  // default context that shared by all GPU devices.
  std::vector<std::unique_ptr<sycl::context>> contexts;
  std::mutex devices_mutex;
} gDevPool;

static void enumDevices(std::vector<std::unique_ptr<sycl::device>>& devices) {
  std::vector<sycl::device> root_devices;
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from GPU platform firstly.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }

  if (!Settings::I().is_device_hierarchy_composite_enabled()) {
    // With `FLAT` and `COMBINED` mode, all sub-devices are retured.
    for (const auto& sub_device : root_devices) {
      devices.push_back(std::make_unique<sycl::device>(sub_device));
    }
    return;
  }

  // For `COMPOSITE` mode, all root-devices are returned. Implicit scaling
  // is allowed. If IPEX_TILE_AS_DEVICE is ON, tile partition is enabled.
  if (Settings::I().is_tile_as_device_enabled()) {
    constexpr sycl::info::partition_property partition_by_affinity =
        sycl::info::partition_property::partition_by_affinity_domain;
    constexpr sycl::info::partition_affinity_domain next_partitionable =
        sycl::info::partition_affinity_domain::next_partitionable;
    for (const auto& root_device : root_devices) {
      try {
        auto sub_devices =
            root_device.create_sub_devices<partition_by_affinity>(
                next_partitionable);
        for (auto& sub_device : sub_devices) {
          devices.push_back(std::make_unique<sycl::device>(sub_device));
        }
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
          TORCH_WARN_ONCE(
              "Tile partition is UNSUPPORTED : ",
              root_device.get_info<dpcpp_dev_name>());
        }
        devices.push_back(std::make_unique<sycl::device>(root_device));
      }
    }
  } else {
    for (const auto& root_device : root_devices) {
      // Tile partition is disabled, all root-devices are returned.
      devices.push_back(std::make_unique<sycl::device>(root_device));
    }
  }
}

// It should be call only once. (std::call_once)
static void initGlobalDevicePoolState() {
  enumDevices(gDevPool.devices);

  auto device_count = gDevPool.devices.size();
  if (device_count <= 0) {
    TORCH_WARN("XPU Device count is zero!");
    return;
  }

#if defined(USE_MULTI_CONTEXT)
  gDevPool.contexts.resize(device_count);
  for (int i = 0; i < device_count; i++) {
    gDevPool.contexts[i] = std::make_unique<sycl::context>(
        sycl::context({*gDevPool.devices[i]}, dpcppAsyncHandler));
  }
#else
  gDevPool.contexts.resize(1);
  gDevPool.contexts[0] = std::make_unique<sycl::context>(
      gDevPool.devices[0]->get_platform().ext_oneapi_get_default_context());
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
    TORCH_CHECK(0, "dpcppSetDevice: device_id is out of range");
  } else {
    cur_dev_index = device_id;
  }
  return DPCPP_SUCCESS;
}

sycl::device& dpcppGetRawDevice(DeviceId device_id) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_id >= (DeviceId)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppGetRawDevice: device_id is out of range");
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

bool dpcppIsDevPoolInit() {
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  return gDevPool.devices.size() > 0;
}

sycl::context& dpcppGetDeviceContext(DeviceId device) {
  initDevicePoolCallOnce();
#if defined(USE_MULTI_CONTEXT)
  DeviceId device_id = device;
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  return *gDevPool.contexts[device_id];
#else
  return *gDevPool.contexts[0];
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
  device_info.resize(num_gpus);
  device_properties.resize(num_gpus);
}

static void initDeviceProperty(DeviceId device_id) {
  DeviceProp device_prop;
  auto& device = dpcppGetRawDevice(device_id);

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
#if (defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20240100)
  device_prop.device_arch = device.get_info<dpcpp_dev_architecture>();
#endif
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
  // According to existing platform, default value 512 is large enough to
  // subscribe a latest modern platform. But on possible platforms, it may get a
  // little perf drop in some cases.
  device_prop.gpu_eu_count = device.has(dpcpp_dev_aspect_gpu_eu_count)
      ? device.get_info<dpcpp_dev_ext_intel_gpu_eu_count>()
      : 512;
  device_prop.gpu_eu_count_per_subslice =
      device.has(dpcpp_dev_aspect_gpu_eu_count_per_subslice)
      ? device.get_info<dpcpp_dev_ext_intel_gpu_eu_count_per_subslice>()
      : 8;
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

  auto convert_dev_type = [&]() {
    switch (device_prop.dev_type) {
      case sycl::info::device_type::cpu:
        return device_type::cpu;
      case sycl::info::device_type::gpu:
        return device_type::gpu;
      case sycl::info::device_type::accelerator:
        return device_type::accelerator;
      case sycl::info::device_type::host:
        return device_type::host;
      default:
        throw std::runtime_error("Unknown/unsupport sycl device type!");
    }
  };

  DeviceInfo dev_info;
  dev_info.dev_type = convert_dev_type();
  dev_info.dev_name = device_prop.dev_name;
  dev_info.platform_name = device_prop.platform_name;
  dev_info.vendor = device_prop.vendor;
  dev_info.driver_version = device_prop.driver_version;
  dev_info.version = device_prop.version;
  dev_info.global_mem_size = device_prop.global_mem_size;
  dev_info.max_compute_units = device_prop.max_compute_units;
  dev_info.gpu_eu_count = device_prop.gpu_eu_count;
  dev_info.gpu_subslice_count =
      device_prop.gpu_eu_count / device_prop.gpu_eu_count_per_subslice;
  dev_info.max_work_group_size = device_prop.max_work_group_size;
  dev_info.max_num_sub_groups = device_prop.max_num_subgroup;
  dev_info.sub_group_sizes = device_prop.subgroup_sizes;
  dev_info.support_fp64 = device_prop.support_fp64;
#if (defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20240100)
  dev_info.device_arch = static_cast<uint64_t>(device_prop.device_arch);
#else
  dev_info.device_arch = (uint64_t)0;
#endif
  device_info[device_id] = dev_info;
}

static inline DeviceId init_device_prop(DeviceId device) {
  std::call_once(init_prop_flag, initDevPropVectors);
  DeviceId device_id = device;
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  auto num_gpus = 0;
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&num_gpus));
  AT_ASSERT(device_id >= 0 && device_id < num_gpus);
  std::call_once(device_prop_flags[device_id], initDeviceProperty, device_id);
  return device_id;
}

DeviceProp* dpcppGetCurrentDeviceProperties() {
  DeviceId device = 0;
  AT_DPCPP_CHECK(dpcppGetDevice(&device));
  return dpcppGetDeviceProperties(device);
}

DeviceProp* dpcppGetDeviceProperties(DeviceId device) {
  return &device_properties[init_device_prop(device)];
}

DeviceInfo* dpcppGetCurrentDeviceInfo() {
  DeviceId device = 0;
  AT_DPCPP_CHECK(dpcppGetDevice(&device));
  return dpcppGetDeviceInfo(device);
}

DeviceInfo* dpcppGetDeviceInfo(DeviceId device_id) {
  return &device_info[init_device_prop(device_id)];
}

/*
 * Runtime in multiprocessing note
 *
 * We have known the limitation of fork support in SYCL runtime and LevelZero
 * runtime. If we call runtime APIs in parent process, then fork a child
 * process, there will be an error in runtime if submit any kernel in parent
 * process or child process.
 *
 * In general, `zeInit` must be called after fork, not before. That's because if
 * it is called before fork some structs are inherited by the child process,
 * which means both child and parent are referring to the same internal
 * structures, producing problems, like on SYCL kernel's submission.
 *
 * So we have to call runtime APIs using another fork, pipe the result back
 * to the parent, and then fork the actual child process. For example, we would
 * like to get device count before fork process to check if XPU device is
 * available.
 *
 * We have to fork another child process before calling `zeInit` API in parent
 * process. Then query device count using SYCL runtime APIs. Finally pipe the
 * result to parent process. Now we can check if XPU device is available and
 * fork the actual child process to do the calculation.
 *
 */

int dpcppGetDeviceCountImpl() noexcept {
  std::vector<std::unique_ptr<sycl::device>> devices;
  enumDevices(devices);
  return devices.size();
}

// Return the number of device in `device_count` with a forking processing.
int dpcppGetDeviceCountFork(int& device_count) noexcept {
#ifndef _WIN32
  std::array<int, 1> buffer;
  std::array<int, 2> pipefd;
  if (pipe(pipefd.data()) != 0) {
    return -1;
  }

  int pid = fork();
  if (pid < 0) {
    return -1;
  } else if (pid == 0) { // child process
    buffer[0] = dpcppGetDeviceCountImpl();
    close(pipefd[0]);
    write(pipefd[1], buffer.data(), sizeof(buffer));
    close(pipefd[1]);
    _exit(0);
  } else { // parent process
    wait(NULL);
    close(pipefd[1]);
    read(pipefd[0], buffer.data(), sizeof(buffer));
    close(pipefd[0]);
  }

  device_count = buffer[0];
  return 0;
#else
  return -1;
#endif
}

int dpcppGetDeviceHasFP64DtypeImpl(
    DeviceId device_id,
    bool& has_fp64) noexcept {
  std::vector<std::unique_ptr<sycl::device>> devices;
  enumDevices(devices);
  auto device_count = devices.size();
  if (device_id == -1) {
    device_id = 0;
  }
  if (device_id < 0 || device_id >= device_count) {
    return -1;
  }
  auto& device = *devices[device_id];
  has_fp64 = device.has(dpcpp_dev_aspect_fp64);
  return 0;
}

// Returns in `has_fp64` if support FP64 data type on device `device_id`
// with a forking processing.
int dpcppGetDeviceHasFP64DtypeFork(int device_id, bool& has_fp64) noexcept {
#ifndef _WIN32
  std::array<int, 2> buffer;
  std::array<int, 2> pipefd;
  if (pipe(pipefd.data()) != 0) {
    return -1;
  }

  int pid = fork();
  if (pid < 0) {
    return -1;
  } else if (pid == 0) { // child process
    bool has_fp64 = false;
    buffer[1] = dpcppGetDeviceHasFP64DtypeImpl(device_id, has_fp64);
    buffer[0] = (int)has_fp64;
    close(pipefd[0]);
    write(pipefd[1], buffer.data(), sizeof(buffer));
    close(pipefd[1]);
    _exit(0);
  } else { // parent process
    wait(NULL);
    close(pipefd[1]);
    read(pipefd[0], buffer.data(), sizeof(buffer));
    close(pipefd[0]);
  }

  has_fp64 = (bool)buffer[0];
  return buffer[1];
#else
  return -1;
#endif
}

// This function can be used to get device count and no execption. It is used in
// device_count() and is_available() such that both two functions can be called
// before forking process.
int dpcppPrefetchDeviceCount(int& device_count) noexcept {
  return xpu::dpcpp::dpcppGetDeviceCountFork(device_count);
}

// This function can be used to get if fp64 data type is supported and no
// execption. It is used in device_count() and is_available() such that both two
// functions can be called before forking process.
int dpcppPrefetchDeviceHasFP64Dtype(int device_id, bool& has_fp64) noexcept {
  return xpu::dpcpp::dpcppGetDeviceHasFP64DtypeFork(device_id, has_fp64);
}

} // namespace dpcpp
} // namespace xpu
