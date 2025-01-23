#include <runtime/Device.h>

#include <vector>

namespace torch_ipex::xpu {
namespace dpcpp {

// XXX: The integrity approach should be querying ISA info whether it
// is supported in the specified platform. Querying `architecture` is a WA since
// it is not feasible to maintain a list of all candidates. We will replace
// the WA with querying ISA info once SYCL runtime supports it.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc#feature-test-macro

#if __INTEL_LLVM_COMPILER < 20240200
// PVC_VG not supported in SYCL architecture of OneAPI 2024.1
static const std::array pvc_vg_device_list = {0xbd4};

// MTL iGPU not supported in SYCL architecture of OneAPI 2024.1
static const std::array mtl_device_list = {0x7D55, 0x7DD5, 0x7D57, 0x7DD7};
#endif

// BMG dGPU not supported in SYCL architecture of OneAPI 2024.1 and 2024.2
static const std::array bmg_device_list = {0xE20B};

// FS1 Coral Simulator not supported in SYCL architecture of OneAPI 2024.1
static const std::array coral_device_list = {0x0b73};

bool dpcppGetDeviceHasXMX(DeviceId device_id) noexcept {
  using namespace sycl::ext;
  using namespace sycl::ext::oneapi;
  sycl::device& device = at::xpu::get_raw_device(device_id);
  if (device.has(sycl::aspect::ext_intel_device_id)) {
    auto ext_intel_device_id =
        device.get_info<intel::info::device::device_id>();
#if __INTEL_LLVM_COMPILER < 20240200
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
#endif
    for (uint32_t coral_device_id : coral_device_list) {
      if (ext_intel_device_id == coral_device_id) {
        return false;
      }
    }
    for (uint32_t bmg_device_id : bmg_device_list) {
      if (ext_intel_device_id == bmg_device_id) {
        return true;
      }
    }
  }

  try {
    auto deviceArch =
        device.get_info<experimental::info::device::architecture>();
    if (deviceArch <= experimental::architecture::intel_gpu_dg1) {
      return false;
    } else {
#if __INTEL_LLVM_COMPILER >= 20240200
      switch (deviceArch) {
        case experimental::architecture::intel_gpu_pvc_vg:
        case experimental::architecture::intel_gpu_mtl_u:
        case experimental::architecture::intel_gpu_mtl_h:
          return false;
        case experimental::architecture::intel_gpu_arl_h:
          return true;
        default:
          return true;
      }
    }
#else
      return true;
    }
#endif
  } catch (sycl::exception) {
    TORCH_WARN_ONCE(
        "Detect an unknown architecture, will treat it as no XMX feature support.");
    return false;
  }
}

bool dpcppGetDeviceHas2DBlock(DeviceId device_id) noexcept {
  using namespace sycl::ext;
  using namespace sycl::ext::oneapi;
  sycl::device& device = at::xpu::get_raw_device(device_id);
  if (device.has(sycl::aspect::ext_intel_device_id)) {
    auto ext_intel_device_id =
        device.get_info<intel::info::device::device_id>();
#if __INTEL_LLVM_COMPILER < 20240200
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
#endif
    for (uint32_t coral_device_id : coral_device_list) {
      if (ext_intel_device_id == coral_device_id) {
        return false;
      }
    }
    for (uint32_t bmg_device_id : bmg_device_list) {
      if (ext_intel_device_id == bmg_device_id) {
        return true;
      }
    }
  }

  try {
    auto deviceArch =
        device.get_info<experimental::info::device::architecture>();
    if (deviceArch <= experimental::architecture::intel_gpu_dg1) {
      return false;
    }
    switch (deviceArch) {
      case experimental::architecture::intel_gpu_dg2_g10:
      case experimental::architecture::intel_gpu_dg2_g11:
      case experimental::architecture::intel_gpu_dg2_g12:
#if __INTEL_LLVM_COMPILER >= 20240200
      case experimental::architecture::intel_gpu_mtl_u:
      case experimental::architecture::intel_gpu_mtl_h:
      case experimental::architecture::intel_gpu_arl_h:
#endif
        return false;
      default:
        return true;
    }
  } catch (sycl::exception) {
    TORCH_WARN_ONCE(
        "Detect an unknown architecture, will treat it as no 2D Block feature support.");
    return false;
  }
}

} // namespace dpcpp
} // namespace torch_ipex::xpu