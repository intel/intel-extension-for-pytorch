#pragma once

#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>
#include "include/common/core/arch_config.hpp"

static const std::array ptl_device_list = {0xB0B0, 0xB082};
static const std::array bmg_device_list = {0xE221, 0xE222, 0xE223};

namespace gpu::xetla {
static inline gpu_arch get_device_gpu_arch() {
  using namespace sycl::ext;
  using namespace sycl::ext::oneapi;

  at::DeviceIndex device_id = at::xpu::current_device();
  sycl::device& device = at::xpu::get_raw_device(device_id);

  if (device.has(sycl::aspect::ext_intel_device_id)) {
    auto ext_intel_device_id =
        device.get_info<intel::info::device::device_id>();
#if __INTEL_LLVM_COMPILER < 20250100
    for (uint32_t ptl_device_id : ptl_device_list) {
      if (ext_intel_device_id == ptl_device_id) {
        return gpu_arch::XeHpc;
      }
    }
#endif
    for (uint32_t bmg_device_id : bmg_device_list) {
      if (ext_intel_device_id == bmg_device_id) {
        return gpu_arch::XeHpc;
      }
    }
  }

  auto deviceArch = device.get_info<experimental::info::device::architecture>();
  switch (deviceArch) {
    case experimental::architecture::intel_gpu_pvc:
      return gpu_arch::XeHpc;
    case experimental::architecture::intel_gpu_bmg_g21:
      return gpu_arch::XeHpc;
      // return gpu_arch::Xe2Hpg; //will diff with XeHpc as need
    case experimental::architecture::intel_gpu_lnl_m:
      return gpu_arch::XeHpc;
      // return gpu_arch::Xe2Lpg;  //will diff with XeHpc as need
#if __INTEL_LLVM_COMPILER >= 20250100
    case experimental::architecture::intel_gpu_ptl_h:
    case experimental::architecture::intel_gpu_ptl_u:
      return gpu_arch::XeHpc;
#endif
#if __INTEL_LLVM_COMPILER >= 20240200
    case experimental::architecture::intel_gpu_pvc_vg:
      return gpu_arch::XeHpc_vg;
#endif
    case experimental::architecture::intel_gpu_dg2_g10:
    case experimental::architecture::intel_gpu_dg2_g11:
    case experimental::architecture::intel_gpu_dg2_g12:
    case experimental::architecture::intel_gpu_arl_h:
      return gpu_arch::XeHpg;
    case experimental::architecture::intel_gpu_mtl_u:
    case experimental::architecture::intel_gpu_mtl_h:
    case experimental::architecture::intel_gpu_dg1:
      return gpu_arch::XeLpg;
    default:
      break;
  }

  auto ext_intel_device_id = device.get_info<intel::info::device::device_id>();
  if (ext_intel_device_id == 0x0b73) {
    return gpu_arch::XeHpc; // coral device
  }

#if __INTEL_LLVM_COMPILER < 20240200
  // PVC_VG not supported in SYCL architecture of OneAPI 2024.1
  if (ext_intel_device_id == 0xbd4) {
    return gpu_arch::XeHpc_vg;
  }
#endif

  return gpu_arch::XeLpg;
}

static inline gpu_arch get_xetla_current_arch_tag() {
  static gpu_arch arch_tag = get_device_gpu_arch();
  return arch_tag;
}

} // namespace gpu::xetla
