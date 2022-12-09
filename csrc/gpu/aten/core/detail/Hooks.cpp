#include <utils/DPCPP.h>

#include <ATen/Config.h>
#include <ATen/Context.h>
#include <c10/util/Exception.h>

#include <core/Allocator.h>
#include <core/Convertor.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <core/detail/Hooks.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace xpu {
namespace dpcpp {
namespace detail {

void XPUHooks::initXPU() const {
  // TODO:
}

bool XPUHooks::hasXPU() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "DPCPP backend version: 1.0";
}

at::Device XPUHooks::getATenDeviceFromDLPackDevice(
    const DLDevice& dl_device,
    void* data) const {
  return getATenDeviceFromUSM(data, dl_device.device_id);
}

DLDevice XPUHooks::getDLPackDeviceFromATenDevice(
    const at::Device& aten_device,
    void* data) const {
  TORCH_CHECK(aten_device.is_xpu(), "Only the XPU device type is expected.");
  sycl::device xpu_dev = xpu::dpcpp::dpcppGetRawDevice(aten_device.index());

  sycl::device parent_root_device;
  if (Settings::I().is_tile_as_device_enabled()) {
    try {
      parent_root_device =
          xpu_dev.get_info<sycl::info::device::parent_device>();
    } catch (sycl::exception e) {
      if (e.code() == sycl::errc::invalid) {
        // Gen9 device without tile?
        parent_root_device = xpu_dev;
      } else {
        throw e;
      }
    }
  } else {
    // the root device is returned directly.
    parent_root_device = xpu_dev;
  }

  // find position of parent_root_device in sycl::get_devices
  auto all_root_devs = sycl::device::get_devices();
  auto beg = std::begin(all_root_devs);
  auto end = std::end(all_root_devs);
  auto selector_fn = [parent_root_device](const sycl::device& root_d) -> bool {
    return parent_root_device == root_d;
  };
  auto pos = find_if(beg, end, selector_fn);

  TORCH_CHECK(pos != end, "Could not produce DLPack: failed finding device_id");
  std::ptrdiff_t dev_idx = std::distance(beg, pos);

  return DLDevice{kDLOneAPI, dev_idx};
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace detail
} // namespace dpcpp
} // namespace xpu
