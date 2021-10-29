#pragma once

#include <cpuid.h>

namespace torch_ipex {
namespace cpu {
namespace utils {

class CPUISA {
 public:
  static CPUISA& info() {
    static CPUISA cpu_isa;
    return cpu_isa;
  }

 public:
  bool does_support_avx2() {
    return (type_ & bit_AVX2) != 0;
  }

  bool does_support_avx512() {
    return ((type_ & bit_AVX512F) != 0) && ((type_ & bit_AVX512DQ) != 0) &&
        ((type_ & bit_AVX512BW) != 0) && ((type_ & bit_AVX512VL) != 0);
  }

 private:
  CPUISA() {
    type_ = 0;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    unsigned int data[4] = {};
    const unsigned int& EAX = data[0];
    const unsigned int& EBX = data[1];

    // EAX=0: Highest Function Parameter and Manufacturer ID
    get_cpu_id(0, data);
    if (EAX >= 7) {
      // This returns extended feature flags in
      // EBX, ECX, and EDX. Returns the maximum ECX value for EAX=7 in EAX.
      // EAX=7, ECX=0: Extended Features
      get_cpu_id_ex(7, 0, data);

      if (EBX & bit_AVX2)
        type_ |= bit_AVX2;
      if (EBX & bit_AVX512F)
        type_ |= bit_AVX512F;
      if (EBX & bit_AVX512DQ)
        type_ |= bit_AVX512DQ;
      if (EBX & bit_AVX512BW)
        type_ |= bit_AVX512BW;
      if (EBX & bit_AVX512VL)
        type_ |= bit_AVX512VL;
    }
  }

  ~CPUISA() = default;
  CPUISA(const CPUISA&) = default;
  CPUISA& operator=(const CPUISA&) = default;

 private:
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  static inline void get_cpu_id(unsigned int eax_in, unsigned int data[4]) {
#ifdef _MSC_VER
    __cpuid(reinterpret_cast<int*>(data), eax_in);
#else
    __cpuid(eax_in, data[0], data[1], data[2], data[3]);
#endif
  }

  static inline void get_cpu_id_ex(
      unsigned int eax_in,
      unsigned int ecx_in,
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      unsigned int data[4]) {
#ifdef _MSC_VER
    __cpuidex(reinterpret_cast<int*>(data), eax_in, ecx_in);
#else
    __cpuid_count(eax_in, ecx_in, data[0], data[1], data[2], data[3]);
#endif
  }

 private:
  uint64_t type_;
};

} // namespace utils
} // namespace cpu
} // namespace torch_ipex
