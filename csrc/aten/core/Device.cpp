#include <runtime/Device.h>
#include <runtime/Exception.h>
#include "LazyInit.h"

using namespace at;

namespace xpu {
namespace dpcpp {

FnPtr lazy_init_callback = nullptr;

void setLazyInit(FnPtr fn) {
  lazy_init_callback = fn;
}

DeviceIndex prefetch_device_count() noexcept {
  int count = dpcppPrefetchDeviceCount();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count() noexcept {
  int count;
  LAZY_INIT_CALLBACK(lazy_init_callback)
  int err = dpcppGetDeviceCount(&count);
  return (err == DPCPP_SUCCESS) ? static_cast<DeviceIndex>(count) : 0;
}

DeviceIndex current_device() {
  DeviceIndex cur_device;
  LAZY_INIT_CALLBACK(lazy_init_callback)
  AT_DPCPP_CHECK(dpcppGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  LAZY_INIT_CALLBACK(lazy_init_callback)
  AT_DPCPP_CHECK(dpcppSetDevice(static_cast<int>(device)));
}

DeviceIndex get_device_index_from_ptr(void* ptr) {
  DeviceIndex device_index;
  LAZY_INIT_CALLBACK(lazy_init_callback)
  AT_DPCPP_CHECK(dpcppGetDeviceIdFromPtr(&device_index, ptr));
  return device_index;
}

DeviceProp* getCurrentDeviceProperties() {
  LAZY_INIT_CALLBACK(lazy_init_callback)
  return dpcppGetCurrentDeviceProperties();
}

DeviceProp* getDeviceProperties(DeviceIndex device) {
  LAZY_INIT_CALLBACK(lazy_init_callback)
  return dpcppGetDeviceProperties(device);
}

std::vector<int> prefetchDeviceIdListForCard(int card_id) {
  return dpcppPrefetchDeviceIdListForCard(card_id);
}

std::vector<int>& getDeviceIdListForCard(int card_id) {
  LAZY_INIT_CALLBACK(lazy_init_callback)
  return dpcppGetDeviceIdListForCard(card_id);
}

} // namespace dpcpp
} // namespace xpu
