#include <runtime/Device.h>
#include <runtime/Exception.h>
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

DeviceIndex prefetch_device_count() noexcept {
  int count = dpcppPrefetchDeviceCount();
  return static_cast<DeviceIndex>(count);
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

DeviceIndex get_device_index_from_ptr(void* ptr) {
  DeviceIndex device_index;
  do_pre_init_hook();
  AT_DPCPP_CHECK(dpcppGetDeviceIdFromPtr(&device_index, ptr));
  return device_index;
}

DeviceProp* getCurrentDeviceProperties() {
  do_pre_init_hook();
  return dpcppGetCurrentDeviceProperties();
}

DeviceProp* getDeviceProperties(DeviceIndex device) {
  do_pre_init_hook();
  return dpcppGetDeviceProperties(device);
}

std::vector<int> prefetchDeviceIdListForCard(int card_id) {
  return dpcppPrefetchDeviceIdListForCard(card_id);
}

std::vector<int>& getDeviceIdListForCard(int card_id) {
  do_pre_init_hook();
  return dpcppGetDeviceIdListForCard(card_id);
}

} // namespace dpcpp
} // namespace xpu
