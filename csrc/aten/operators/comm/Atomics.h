#pragma once

#include <c10/util/complex.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "Numerics.h"
#include "Scalar.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<float>& address,
    float val) {
  dpcpp_atomic_ref_relaxed_t<float> target(*address);
  target.fetch_add(val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<double>& address,
    double val) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long assumed = *address_as_ull;
  unsigned long long newval;
  dpcpp_atomic_ref_relaxed_t<unsigned long long> target(*address_as_ull);

  do {
    newval = __double_as_long_long(val + __long_long_as_double(assumed));
  } while (!target.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<at::Half>& address,
    at::Half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int assumed = *address_as_ui;
  unsigned int newval;
  dpcpp_atomic_ref_relaxed_t<unsigned int> target(*address_as_ui);

  do {
    newval = assumed;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
    hsum = Numerics<at::Half>::add(hsum, val);
    newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                                 : (newval & 0xffff0000) | hsum.x;
  } while (!target.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<at::BFloat16>& address,
    at::BFloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int assumed = *address_as_ui;
  unsigned int newval;
  dpcpp_atomic_ref_relaxed_t<unsigned int> target(*address_as_ui);

  do {
    newval = assumed;
    at::BFloat16 hsum;
    hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
    hsum = Numerics<at::BFloat16>::add(hsum, val);
    newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                                 : (newval & 0xffff0000) | hsum.x;
  } while (!target.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int>& address,
    int val) {
  dpcpp_atomic_ref_relaxed_t<int> target(*address);
  target.fetch_add(val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<long>& address,
    int val) {
  dpcpp_atomic_ref_relaxed_t<long> target(*address);
  target.fetch_add(val);
}

template <typename T, size_t n>
struct AtomicAddIntegerImpl;

template <typename T>
struct AtomicAddIntegerImpl<T, 1> {
  inline DPCPP_DEVICE void operator()(T* address, T val) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t assumed = *address_as_ui;
    uint32_t shift = offset * 8;
    uint32_t newval;
    uint32_t newval_byte;

    dpcpp_atomic_ref_relaxed_t<uint32_t> target(*address_as_ui);
    do {
      newval = assumed;
      newval_byte = (newval >> shift) & 0xff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint8_t>(Numerics<int8_t>::add(val, newval_byte));
      newval = (assumed & ~(0x000000ff << shift)) | (newval << shift);
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicAddIntegerImpl<T, 2> {
  inline DPCPP_DEVICE void operator()(T* address, T val) {
    size_t offset = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool is_32_align = offset;
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    uint32_t newval_bytes;
    dpcpp_atomic_ref_relaxed_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_bytes = is_32_align ? newval >> 16 : newval & 0xffff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint16_t>(Numerics<int16_t>::add(val, newval_bytes));
      newval = is_32_align ? (assumed & 0xffff) | (newval << 16)
                           : (assumed & 0xffff0000) | newval;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<uint8_t>& address,
    uint8_t val) {
  AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int8_t>& address,
    int8_t val) {
  AtomicAddIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int16_t>& address,
    int16_t val) {
  AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<bool>& address,
    bool val) {
  *address = address && val;
}

template <typename T>
static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(&address->real_, val.real_);
  atomicAdd(&address->imag_, val.imag_);
}

} // namespace AtenIpexTypeXPU
} // namespace at

// (TODO) add support for atomicAdd
// dpcpp_global_ptr_pt<uint8_t *>
// dpcpp_global_ptr_pt<int8_t>
// dpcpp_global_ptr_pt<bool *>
// dpcpp_global_ptr_pt<int64_t>
// dpcpp_global_ptr_pt<double *>
