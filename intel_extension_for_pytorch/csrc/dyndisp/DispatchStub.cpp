#include "DispatchStub.h"

#include <c10/util/Exception.h>

#include "../cpu/isa/cpu_feature.hpp"

#include <cstdlib>
#include <cstring>

namespace torch_ipex {
namespace cpu {

static CPUCapability compute_cpu_capability() {
  /*
  IPEX also allign to pytorch config environment, and keep the same behavior.
  */
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
    if (strcmp(envar, "avx512") == 0) {
      return CPUCapability::AVX512;
    }
    if (strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }

    if (strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
  }

  if (CPUFeature::get_instance().os_avx512() &&
      CPUFeature::get_instance().cpuid_avx512_vl() &&
      CPUFeature::get_instance().cpuid_avx512_bw() &&
      CPUFeature::get_instance().cpuid_avx512_dq() &&
      CPUFeature::get_instance().cpuid_avx512_f()) {
    // CHECK_SSE(C "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw
    // -mfma;/arch:AVX512") CHECK_SSE(CXX "AVX512" " ;-mavx512f -mavx512dq
    // -mavx512vl -mavx512bw -mfma;/arch:AVX512")
    return CPUCapability::AVX512;
  }
  if (CPUFeature::get_instance().os_avx2() &&
      CPUFeature::get_instance().cpuid_avx2()) {
    // CHECK_SSE(C "AVX2" " ;-mavx2 -mfma;/arch:AVX2")
    // CHECK_SSE(CXX "AVX2" " ;-mavx2 -mfma;/arch:AVX2")
    return CPUCapability::AVX2;
  }

  return CPUCapability::DEFAULT;
}

CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

void* DispatchStubImpl::get_call_ptr(
    DeviceType device_type,
    void* DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    ,
    void* AVX512
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
#ifdef HAVE_AVX512_CPU_DEFINITION
            ,
            AVX512
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

    case DeviceType::CUDA:
      TORCH_INTERNAL_ASSERT(
          cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
      return cuda_dispatch_ptr;

    case DeviceType::HIP:
      TORCH_INTERNAL_ASSERT(
          hip_dispatch_ptr, "DispatchStub: missing HIP kernel");
      return hip_dispatch_ptr;

    default:
      AT_ERROR("DispatchStub: unsupported device type", device_type);
  }
}

void* DispatchStubImpl::choose_cpu_impl(
    void* DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    ,
    void* AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    ,
    void* AVX2
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
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
