#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>

using namespace at;

namespace xpu {
namespace dpcpp {

DeviceIndex prefetch_device_count() noexcept {
  int count = dpcppPrefetchDeviceCount();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count() noexcept {
  int count;
  lazy_init();
  int err = dpcppGetDeviceCount(&count);
  return (err == DPCPP_SUCCESS) ? static_cast<DeviceIndex>(count) : 0;
}

DeviceIndex current_device() {
  DeviceIndex cur_device;
  lazy_init();
  AT_DPCPP_CHECK(dpcppGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  lazy_init();
  AT_DPCPP_CHECK(dpcppSetDevice(static_cast<int>(device)));
}

DeviceIndex get_device_index_from_ptr(void* ptr) {
  DeviceIndex device_index;
  lazy_init();
  AT_DPCPP_CHECK(dpcppGetDeviceIdFromPtr(&device_index, ptr));
  return device_index;
}

DeviceProp* getCurrentDeviceProperties() {
  lazy_init();
  return dpcppGetCurrentDeviceProperties();
}

DeviceProp* getDeviceProperties(DeviceIndex device) {
  lazy_init();
  return dpcppGetDeviceProperties(device);
}

std::vector<int> prefetchDeviceIdListForCard(int card_id) {
  return dpcppPrefetchDeviceIdListForCard(card_id);
}

std::vector<int>& getDeviceIdListForCard(int card_id) {
  lazy_init();
  return dpcppGetDeviceIdListForCard(card_id);
}

} // namespace dpcpp
} // namespace xpu
