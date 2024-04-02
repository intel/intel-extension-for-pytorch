#include "DispatchStub.h"

#include <c10/util/Exception.h>

#include "../cpu/isa/cpu_feature.hpp"

#include <dnnl.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace torch_ipex {
namespace cpu {

using namespace dnnl;

const char* CPUCapabilityToString(CPUCapability isa) {
  switch (isa) {
    case CPUCapability::DEFAULT:
      return "DEFAULT";
    case CPUCapability::AVX2:
      return "AVX2";
    case CPUCapability::AVX2_VNNI:
      return "AVX2_VNNI";
    case CPUCapability::AVX512:
      return "AVX512";
    case CPUCapability::AVX512_VNNI:
      return "AVX512_VNNI";
    case CPUCapability::AVX512_BF16:
      return "AVX512_BF16";
    case CPUCapability::AMX:
      return "AMX";
    case CPUCapability::AVX512_FP16:
      return "AVX512_FP16";
    case CPUCapability::NUM_OPTIONS:
      return "OutOfBoundaryLevel";

    default:
      return "WrongLevel";
  }
}

CPUCapability _get_highest_cpu_support_isa_level() {
  /*
  reference to FindAVX.cmake
  */
  if (CPUFeature::get_instance().isa_level_avx512_fp16()) {
    return CPUCapability::AVX512_FP16;
  } else if (CPUFeature::get_instance().isa_level_amx()) {
    return CPUCapability::AMX;
  } else if (CPUFeature::get_instance().isa_level_avx512_bf16()) {
    return CPUCapability::AVX512_BF16;
  } else if (CPUFeature::get_instance().isa_level_avx512_vnni()) {
    return CPUCapability::AVX512_VNNI;
  } else if (CPUFeature::get_instance().isa_level_avx512()) {
    return CPUCapability::AVX512;
  }
  if (CPUFeature::get_instance().isa_level_avx2_vnni()) {
    return CPUCapability::AVX2_VNNI;
  } else if (CPUFeature::get_instance().isa_level_avx2()) {
    return CPUCapability::AVX2;
  }

  return CPUCapability::DEFAULT;
}

CPUCapability _get_highest_binary_support_isa_level() {
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
  return CPUCapability::AVX512_FP16;
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
  return CPUCapability::AMX;
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
  return CPUCapability::AVX512_BF16;
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
  return CPUCapability::AVX512_VNNI;
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
  return CPUCapability::AVX512;
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
  return CPUCapability::AVX2_VNNI;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  return CPUCapability::AVX2;
#endif
  return CPUCapability::DEFAULT;
}

static bool _load_sync_onednn_isa_setting() {
  // _IPEX_NOT_SYNC_ONEDNN_ISA is debug only env.
  auto envar = std::getenv("_IPEX_NOT_SYNC_ONEDNN_ISA");
  if (envar) {
    if (strcmp(envar, "1") == 0) {
      return true;
    }
  }
  return false;
}

bool check_not_sync_onednn_isa_level() {
  static bool b_not_sync = _load_sync_onednn_isa_setting();

  return b_not_sync;
}

status set_current_cpu_isa_level_to_onednn(cpu_isa isa) {
  if (check_not_sync_onednn_isa_level()) {
    return status::unimplemented;
  }

  return set_max_cpu_isa(isa);
}

cpu_isa ipex_isa_to_onednn_isa(CPUCapability ipex_isa) {
  switch (ipex_isa) {
    case CPUCapability::DEFAULT:
      return cpu_isa::isa_default;
    case CPUCapability::AVX2:
      return cpu_isa::avx2;
    case CPUCapability::AVX2_VNNI:
      return cpu_isa::avx2_vnni;
    case CPUCapability::AVX512:
      return cpu_isa::avx512_core;
    case CPUCapability::AVX512_VNNI:
      return cpu_isa::avx512_core_vnni;
    case CPUCapability::AVX512_BF16:
      return cpu_isa::avx512_core_bf16;
    case CPUCapability::AMX:
      return cpu_isa::avx512_core_amx;
    case CPUCapability::AVX512_FP16:
      return cpu_isa::avx512_core_amx_fp16;
    case CPUCapability::NUM_OPTIONS:
      TORCH_WARN("DispatchStub: OutOfBoundaryISALevel for IPEX");
      return cpu_isa::isa_default;

    default:
      auto ipex_isa_str = CPUCapabilityToString(ipex_isa);
      auto msg = c10::str(
          "DispatchStub: No corresponding onednn isa for ",
          ipex_isa_str,
          "Please consider check whether this ISA is supported by oneDNN");
      TORCH_WARN(msg);
      return cpu_isa::isa_default;
  }
}

static CPUCapability compute_cpu_capability() {
  CPUCapability highest_cpu_supported_isa_level =
      _get_highest_cpu_support_isa_level();

  CPUCapability highest_binary_supported_isa_level =
      _get_highest_binary_support_isa_level();

  bool b_manual_setup = true;
  CPUCapability manual_setup_isa_level;

  /*
  IPEX also align to pytorch config environment, and keep the same behavior.
  */
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
    if (strcmp(envar, "avx512_fp16") == 0) {
      manual_setup_isa_level = CPUCapability::AVX512_FP16;
    } else if (strcmp(envar, "amx") == 0) {
      manual_setup_isa_level = CPUCapability::AMX;
    } else if (strcmp(envar, "avx512_bf16") == 0) {
      manual_setup_isa_level = CPUCapability::AVX512_BF16;
    } else if (strcmp(envar, "avx512_vnni") == 0) {
      manual_setup_isa_level = CPUCapability::AVX512_VNNI;
    } else if (strcmp(envar, "avx512") == 0) {
      manual_setup_isa_level = CPUCapability::AVX512;
    } else if (strcmp(envar, "avx2_vnni") == 0) {
      manual_setup_isa_level = CPUCapability::AVX2_VNNI;
    } else if (strcmp(envar, "avx2") == 0) {
      manual_setup_isa_level = CPUCapability::AVX2;
    } else if (strcmp(envar, "default") == 0) {
      manual_setup_isa_level = CPUCapability::DEFAULT;
    } else {
      TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
      b_manual_setup = false;
    }
  } else {
    b_manual_setup = false;
  }

  CPUCapability max_support_isa_level = std::min(
      highest_cpu_supported_isa_level, highest_binary_supported_isa_level);
  if (b_manual_setup) {
    cpu_isa manual_onednn_isa = ipex_isa_to_onednn_isa(manual_setup_isa_level);
    set_current_cpu_isa_level_to_onednn(manual_onednn_isa);

    if (manual_setup_isa_level <= max_support_isa_level) {
      return manual_setup_isa_level;
    }
  }

  return max_support_isa_level;
}

// Use global variable to trigger cpu capability initialization, when module
// load.
static CPUCapability g_cpu_capability = compute_cpu_capability();

CPUCapability get_cpu_capability() {
  return g_cpu_capability;
}

void* DispatchStubImpl::get_call_ptr(
    DeviceType device_type,
    void* DEFAULT
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
    ,
    void* AVX512_FP16
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
    ,
    void* AMX
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
    ,
    void* AVX512_BF16
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
    ,
    void* AVX512_VNNI
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
    ,
    void* AVX512
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
    ,
    void* AVX2_VNNI
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    ,
    void* AVX2
#endif
) {
  switch (device_type) {
    case DeviceType::CPU: {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        fptr = choose_cpu_impl(
            DEFAULT
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
            ,
            AVX512_FP16
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
            ,
            AMX
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
            ,
            AVX512_BF16
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
            ,
            AVX512_VNNI
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
            ,
            AVX512
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
            ,
            AVX2_VNNI
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
            ,
            AVX2
#endif
        );
        cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
      }
      return fptr;
    }

      /*
      case DeviceType::XPU:
        TORCH_INTERNAL_ASSERT(
            xpu_dispatch_ptr, "DispatchStub: missing XPU kernel");
        return xpu_dispatch_ptr;
      */

    default:
      AT_ERROR("DispatchStub: unsupported device type", device_type);
  }
}

void* DispatchStubImpl::choose_cpu_impl(
    void* DEFAULT
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
    ,
    void* AVX512_FP16
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
    ,
    void* AMX
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
    ,
    void* AVX512_BF16
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
    ,
    void* AVX512_VNNI
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
    ,
    void* AVX512
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
    ,
    void* AVX2_VNNI
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    ,
    void* AVX2
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512_FP16)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512_FP16)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512_FP16;
    }
  }
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AMX)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AMX)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AMX;
    }
  }
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512_BF16)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512_BF16)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512_BF16;
    }
  }
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512_VNNI)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512_VNNI)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512_VNNI;
    }
  }
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512;
    }
  }
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2_VNNI)) {
    TORCH_INTERNAL_ASSERT(AVX2_VNNI, "DispatchStub: missing AVX2_VNNI kernel");
    return AVX2_VNNI;
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
    return AVX2;
  }
#endif

  TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
  return DEFAULT;
}

} // namespace cpu
} // namespace torch_ipex
