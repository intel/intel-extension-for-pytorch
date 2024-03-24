#include <runtime/Device.h>

#include <vector>

namespace torch_ipex::xpu {
namespace dpcpp {

// XXX: The integrity approach should be querying ISA info whether it
// is supported in the specified platform. Querying `architecture` is a WA since
// it is not feasible to maintain a list of all candidates. We will replace
// the WA with querying ISA info once SYCL runtime supports it.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc#feature-test-macro

// PVC_VG not supported in SYCL architecture of OneAPI 2024.1
static const std::array pvc_vg_device_list = {0xbd4};

// MTL iGPU not supported in SYCL architecture of OneAPI 2024.1
static const std::array mtl_device_list = {0x7D55, 0x7DD5, 0x7D57, 0x7DD7};

// FS1 Coral Simulator not supported in SYCL architecture of OneAPI 2024.1
static const std::array coral_device_list = {0x0b73};

bool dpcppGetDeviceHasXMX(DeviceId device_id) noexcept {
  using namespace sycl::ext;
  using namespace sycl::ext::oneapi;
  sycl::device& device = at::xpu::get_raw_device(device_id);
  auto ext_intel_device_id = device.get_info<intel::info::device::device_id>();
  for (uint32_t pvc_vg_device_id : pvc_vg_device_list) {
    if (ext_intel_device_id == pvc_vg_device_id) {
      return false;
    }
  }
  for (uint32_t mtl_device_id : mtl_device_list) {
    if (ext_intel_device_id == mtl_device_id) {
      return false;
    }
  }
  for (uint32_t coral_device_id : coral_device_list) {
    if (ext_intel_device_id == coral_device_id) {
      return false;
    }
  }
  auto deviceArch = device.get_info<experimental::info::device::architecture>();
  if (deviceArch <= experimental::architecture::intel_gpu_dg1) {
    return false;
  } else {
    // currently PVC and DG2 all support XMX, will update after PVC_VG and MTL
    return true;
  }
}

bool dpcppGetDeviceHas2DBlock(DeviceId device_id) noexcept {
  using namespace sycl::ext;
  using namespace sycl::ext::oneapi;
  sycl::device& device = at::xpu::get_raw_device(device_id);
  auto ext_intel_device_id = device.get_info<intel::info::device::device_id>();
  for (uint32_t pvc_vg_device_id : pvc_vg_device_list) {
    if (ext_intel_device_id == pvc_vg_device_id) {
      return true;
    }
  }
  for (uint32_t mtl_device_id : mtl_device_list) {
    if (ext_intel_device_id == mtl_device_id) {
      return false;
    }
  }
  for (uint32_t coral_device_id : coral_device_list) {
    if (ext_intel_device_id == coral_device_id) {
      return false;
    }
  }
  auto deviceArch = device.get_info<experimental::info::device::architecture>();
  if (deviceArch <= experimental::architecture::intel_gpu_dg1) {
    return false;
  }
  switch (deviceArch) {
    case experimental::architecture::intel_gpu_dg2_g10:
    case experimental::architecture::intel_gpu_dg2_g11:
    case experimental::architecture::intel_gpu_dg2_g12:
      return false;
    default:
      return true;
  }
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
