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
#include <utils/Settings.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace xpu {
namespace dpcpp {
namespace detail {
namespace {

// Cache the global id of each aten device to avoid the SYCL runtime overhead
// for better performance.
static std::vector<int32_t> aten_device_global_ids;
static std::once_flag init_global_id_pool_flag;
static std::deque<std::once_flag> init_aten_device_flag;

static void initGlobalIdPoolState() {
  auto num_gpus = xpu::dpcpp::device_count();
  aten_device_global_ids.resize(num_gpus);
  init_aten_device_flag.resize(num_gpus);
}

// Get gpu device id from all SYCL devices associated with the SYCL platform.
static void initGlobalIdFromATenDeviceId(const DeviceId aten_device_id) {
  sycl::device& xpu_device = xpu::dpcpp::dpcppGetRawDevice(aten_device_id);
  sycl::device target_device;

  if (Settings::I().is_device_hierarchy_composite_enabled()) {
    // In `COMPOSITE` mode, we need to check if the tile partition feature is
    // enabled. If yes, we have to find its parent device.
    if (Settings::I().is_tile_as_device_enabled()) {
      try {
        target_device =
            xpu_device.get_info<sycl::info::device::parent_device>();
      } catch (sycl::exception& e) {
        if (e.code() == sycl::errc::invalid) {
          // Tile partition is not supported.
          target_device = xpu_device;
        } else {
          throw e;
        }
      }
    } else {
      // In `COMPOSITE` mode, here is a root-device if tile partition feature is
      // disabled.
      target_device = xpu_device;
    }
  } else {
    // With `FLAT` and `COMBINED` mode, all sub-devices are exposed. So we don't
    // need to find the parent device anymore and directly find position of sycl
    // device `xpu_device` in sycl::device::get_devices()
    target_device = xpu_device;
  }
  // Get all SYCL devices associated with the SYCL platform.
  auto device_list = sycl::device::get_devices();
  auto beg = std::begin(device_list);
  auto end = std::end(device_list);
  auto selector_fn = [target_device](const sycl::device& device) -> bool {
    return target_device == device;
  };

  auto pos = find_if(beg, end, selector_fn);
  TORCH_CHECK(pos != end, "Failed to find the global id of this aten device.");

  aten_device_global_ids[aten_device_id] =
      static_cast<int32_t>(std::distance(beg, pos));
}
} // anonymous namespace

void XPUHooks::initXPU() const {
  // TODO:
}

bool XPUHooks::hasXPU() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "XPU backend version: 1.0";
}

at::Device XPUHooks::getATenDeviceFromDLPackDevice(
    const DLDevice_& dl_device,
    void* data) const {
  return getATenDeviceFromUSM(data, dl_device.device_id);
}

DLDevice_& XPUHooks::getDLPackDeviceFromATenDevice(
    DLDevice_& dl_device,
    const at::Device& aten_device,
    void* data) const {
  TORCH_CHECK(aten_device.is_xpu(), "Only the XPU device type is expected.");

  auto aten_device_id = aten_device.index();
  // This is a thread-safe implementation via std::call_once.
  std::call_once(init_global_id_pool_flag, initGlobalIdPoolState);
  std::call_once(
      init_aten_device_flag[aten_device_id],
      initGlobalIdFromATenDeviceId,
      aten_device_id);

  auto device_global_id = aten_device_global_ids[aten_device_id];
  dl_device = {kDLOneAPI, device_global_id};

  return dl_device;
}

Generator XPUHooks::getXPUGenerator(DeviceIndex device_index) const {
  auto generator = make_generator<xpu::dpcpp::DPCPPGeneratorImpl>(device_index);
  return generator;
}

const Generator& XPUHooks::getDefaultXPUGenerator(
    DeviceIndex device_index) const {
  const auto& generator = getDefaultDPCPPGenerator(device_index);
  return generator;
}

int XPUHooks::getNumGPUs() const {
  return xpu::dpcpp::device_count();
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace detail
} // namespace dpcpp
} // namespace xpu
