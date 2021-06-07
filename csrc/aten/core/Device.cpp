#include "Device.h"


namespace xpu {

namespace dpcpp {

static std::once_flag init_flag;
static std::deque<std::once_flag> device_flags;
static std::vector<XPUDeviceProp> device_properties;

XPUDeviceProp* getCurrentDeviceProperties() {
  auto device = current_device();
  return getDeviceProperties(device);
}

void initXPUContextVectors() {
  auto num_gpus = device_count();
  device_flags.resize(num_gpus);
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
  std::call_once(init_flag, initXPUContextVectors);
  if (device == -1) device = current_device();
  auto num_gpus = device_count();
  AT_ASSERT(device >= 0 && device < num_gpus);
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

} // namespace dpcpp
} // namespace xpu
