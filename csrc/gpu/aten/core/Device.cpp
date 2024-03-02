#include <core/Device.h>
#include <runtime/Device.h>
#include <runtime/DeviceProp.h>
#include <runtime/Exception.h>
#include <utils/DPCPP.h>
#include "PreInitHook.h"

using namespace at;

namespace xpu {
namespace dpcpp {

static InitFnPtr pre_init_hook = nullptr;

// Here is a hook mechanism that can call lazy_init().
void do_pre_init_hook() {
  // Don't call pre_init_hook when it is nullptr, which means calling lazy_init
  // is unnecessary. It makes sure back-end library libintel-ext-pt-gpu.so can
  // be used independently.
  if (pre_init_hook) {
    pre_init_hook();
  }
}

// Here is a callback pointer that can register lazy_init to pre_init_hook,
// while we can call lazy_init via calling do_pre_init_hook()
void set_pre_init_hook_fn(InitFnPtr fn) {
  pre_init_hook = fn;
}

DeviceIndex device_count_impl() {
  int count;
  int err = dpcppGetDeviceCount(&count);
  return (err == DPCPP_SUCCESS) ? static_cast<DeviceIndex>(count) : 0;
}

DeviceIndex device_count() noexcept {
  do_pre_init_hook();
  // initialize number of devices only once
  static DeviceIndex count = []() {
    try {
      return device_count_impl();
    } catch (std::runtime_error& err) {
      // such as "Failed to apply tile partition"
      TORCH_WARN("XPU initialization: ", err.what());
      return static_cast<DeviceIndex>(0);
    }
  }();
  return count;
}

DeviceIndex current_device() {
  DeviceIndex cur_device;
  do_pre_init_hook();
  AT_DPCPP_CHECK(dpcppGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  do_pre_init_hook();
  AT_DPCPP_CHECK(dpcppSetDevice(static_cast<int>(device)));
}

void* sycl_device(DeviceIndex device) {
  do_pre_init_hook();
  return reinterpret_cast<void*>(&dpcppGetRawDevice(device));
}

DeviceIndex get_device_index_from_ptr(void* ptr) {
  DeviceIndex device_index;
  do_pre_init_hook();
  AT_DPCPP_CHECK(dpcppGetDeviceIdFromPtr(&device_index, ptr));
  return device_index;
}

DeviceInfo* getCurrentDeviceInfo() {
  do_pre_init_hook();
  return dpcppGetCurrentDeviceInfo();
}

DeviceInfo* getDeviceInfo(DeviceIndex device) {
  do_pre_init_hook();
  return dpcppGetDeviceInfo(device);
}

int prefetch_device_count(int& device_count) noexcept {
  return xpu::dpcpp::dpcppPrefetchDeviceCount(device_count);
}

// This function can be used to get if fp64 data type is supported and no
// execption. It is used in device_count() and is_available() such that both two
// functions can be called before forking process.
int prefetch_device_has_fp64_dtype(int device_id, bool& has_fp64) noexcept {
  return xpu::dpcpp::dpcppPrefetchDeviceHasFP64Dtype(device_id, has_fp64);
}

uint64_t getDeviceFreeMemory(DeviceIndex device_id) {
  return xpu::dpcpp::dpcppGetDeviceFreeMemory(device_id);
}

} // namespace dpcpp
} // namespace xpu
