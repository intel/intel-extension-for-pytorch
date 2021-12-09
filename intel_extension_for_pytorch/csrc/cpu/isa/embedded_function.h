#pragma once
#include <stdint.h>

#if defined(__GNUC__)
#include <cpuid.h>
#define __forceinline __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

/* get value from bits [n:m] */
#define BIT_M_TO_N(x, m, n) ((uint32_t)(x << (31 - (n))) >> ((31 - (n)) + (m)))
#define BIT_M_TO_N_64(x, m, n) \
  ((uint64_t)(x << (63 - (n))) >> ((63 - (n)) + (m)))

__forceinline bool check_reg_bit(uint32_t reg, int bit_idx) {
  return (reg & ((uint32_t)1 << bit_idx));
}

__forceinline void read_cpuid(
    uint32_t func_id,
    uint32_t* p_eax,
    uint32_t* p_ebx,
    uint32_t* p_ecx,
    uint32_t* p_edx) {
  int reg_data[4] = {0};
#if defined(__GNUC__)
  __cpuid(func_id, reg_data[0], reg_data[1], reg_data[2], reg_data[3]);
#elif defined(_MSC_VER)
  __cpuid(reg_data, func_id);
#endif
  *p_eax = reg_data[0];
  *p_ebx = reg_data[1];
  *p_ecx = reg_data[2];
  *p_edx = reg_data[3];
}

__forceinline void read_cpuidex(
    uint32_t func_id,
    uint32_t sub_func_id,
    uint32_t* p_eax,
    uint32_t* p_ebx,
    uint32_t* p_ecx,
    uint32_t* p_edx) {
  int reg_data[4] = {0};
#if defined(__GNUC__)
  __cpuid_count(
      func_id, sub_func_id, reg_data[0], reg_data[1], reg_data[2], reg_data[3]);
#elif defined(_MSC_VER)
  __cpuidex(reg_data, func_id, sub_func_id);
#endif
  *p_eax = reg_data[0];
  *p_ebx = reg_data[1];
  *p_ecx = reg_data[2];
  *p_edx = reg_data[3];
}

#define _XCR_XFEATURE_ENABLED_MASK 0

__forceinline bool read_xcr(uint32_t ext_ctrl_reg, uint64_t* p_xcr) {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  /*
  Processor Extended State Enumeration Sub-leaf (EAX = 0DH, ECX = 1)
  EAX:
      Bit 02: Supports XGETBV with ECX = 1 if set.
  */
  read_cpuid(0, &eax, &ebx, &ecx, &edx);
  uint32_t max_basic_id = eax;
  if (max_basic_id < 0x0000000D) {
    return false;
  }

  read_cpuidex(0x0000000D, 1, &eax, &ebx, &ecx, &edx);
  if (check_reg_bit(eax, 2) == false) {
    return false;
  }

  /*
  NPUT EAX = 01H: Returns Feature Information in ECX and EDX
  ECX:
  BIT 27: OSXSAVE, A value of 1 indicates that the OS has set CR4.OSXSAVE[bit
  18] to enable XSETBV/XGETBV instructions to access XCR
  */
  read_cpuid(1, &eax, &ebx, &ecx, &edx);
  if (check_reg_bit(ecx, 27) == false) {
    return false;
  }

#if defined(__GNUC__)
  uint32_t low, high;
  __asm__(".byte 0x0F, 0x01, 0xD0" : "=a"(low), "=d"(high) : "c"(ext_ctrl_reg));
  *p_xcr = ((uint64_t)high << 32) | (uint64_t)low;
#elif defined(_MSC_VER)
  *p_xcr = (uint64_t)_xgetbv((unsigned int)ext_ctrl_reg);
#endif
  return true;
}