#pragma once

#include <c10/util/complex.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "Numerics.h"
#include "Scalar.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename T, size_t n>
struct AtomicIntegerImpl;

template <typename T>
struct AtomicIntegerImpl<T, 1> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(T* address, T val, const func_t& func) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t assumed = *address_as_ui;
    uint32_t shift = offset * 8;
    uint32_t newval;
    uint32_t newval_byte;
    dpcpp_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_byte = (newval >> shift) & 0xff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint8_t>(func(val, static_cast<T>(newval_byte)));
      newval = (assumed & ~(0x000000ff << shift)) | (newval << shift);
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImpl<T, 2> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(T* address, T val, const func_t& func) {
    size_t offset = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool is_32_align = offset;
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    uint32_t newval_bytes;
    dpcpp_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_bytes = is_32_align ? newval >> 16 : newval & 0xffff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint16_t>(func(val, static_cast<T>(newval_bytes)));
      newval = is_32_align ? (assumed & 0xffff) | (newval << 16)
                           : (assumed & 0xffff0000) | newval;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImpl<T, 4> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(T* address, T val, const func_t& func) {
    uint32_t* address_as_ui = (uint32_t*)(address);
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    dpcpp_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = static_cast<uint32_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImpl<T, 8> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(T* address, T val, const func_t& func) {
    unsigned long long* address_as_ull = (unsigned long long*)(address);
    unsigned long long assumed = *address_as_ull;
    unsigned long long newval;
    dpcpp_atomic_ref_rlx_dev_global_t<unsigned long long> target(
        *address_as_ull);

    do {
      newval = static_cast<uint64_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

#define SYCL_ATOMIC_INTEGER(NAME, OP, DTYPE)                  \
  static inline DPCPP_DEVICE void atomic##NAME(               \
      const dpcpp_global_ptr_pt<DTYPE>& address, DTYPE val) { \
    AtomicIntegerImpl<DTYPE, sizeof(DTYPE)>()(                \
        address, val, [](DTYPE a, DTYPE b) { return OP; });   \
  }

template <typename T>
struct AtomicFPImpl;

template <>
struct AtomicFPImpl<at::Half> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(
      at::Half* address,
      at::Half val,
      const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    dpcpp_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = assumed;
      at::Half hsum;
      hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
      hsum = func(hsum, val);
      newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                                   : (newval & 0xffff0000) | hsum.x;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <>
struct AtomicFPImpl<at::BFloat16> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(
      at::BFloat16* address,
      at::BFloat16 val,
      const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    dpcpp_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = assumed;
      at::BFloat16 bsum;
      bsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
      bsum = func(bsum, val);
      newval = (size_t)address & 2 ? (newval & 0xffff) | (bsum.x << 16)
                                   : (newval & 0xffff0000) | bsum.x;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <>
struct AtomicFPImpl<float> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(
      float* address,
      float val,
      const func_t& func) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    dpcpp_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = __float_as_int(func(val, __int_as_float(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <>
struct AtomicFPImpl<double> {
  template <typename func_t>
  inline DPCPP_DEVICE void operator()(
      double* address,
      double val,
      const func_t& func) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long assumed = *address_as_ull;
    unsigned long long newval;
    dpcpp_atomic_ref_rlx_dev_global_t<unsigned long long> target(
        *address_as_ull);

    do {
      newval = __double_as_long_long(func(val, __long_long_as_double(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

#define SYCL_ATOMIC_FP(NAME, OP, DTYPE)                                       \
  static inline DPCPP_DEVICE void atomic##NAME(                               \
      const dpcpp_global_ptr_pt<DTYPE>& address, DTYPE val) {                 \
    AtomicFPImpl<DTYPE>()(address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

// Atomic addition implementation.

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<float>& address,
    float val) {
  dpcpp_atomic_ref_rlx_dev_global_t<float> target(*address);
  target.fetch_add(val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<double>& address,
    double val) {
  dpcpp_atomic_ref_rlx_dev_global_t<double> target(*address);
  target.fetch_add(val);
}

SYCL_ATOMIC_FP(Add, Numerics<at::Half>::add(a, b), at::Half)
SYCL_ATOMIC_FP(Add, Numerics<at::BFloat16>::add(a, b), at::BFloat16)

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int>& address,
    int val) {
  dpcpp_atomic_ref_rlx_dev_global_t<int> target(*address);
  target.fetch_add(val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int64_t>& address,
    int64_t val) {
  dpcpp_atomic_ref_rlx_dev_global_t<int64_t> target(*address);
  target.fetch_add(val);
}

SYCL_ATOMIC_INTEGER(Add, Numerics<uint8_t>::add(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Add, Numerics<int8_t>::add(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Add, Numerics<int16_t>::add(a, b), int16_t)

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<bool>& address,
    bool val) {
  *address = address && val;
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_local_ptr_pt<uint32_t>& address,
    uint32_t val) {
  dpcpp_atomic_ref_rlx_wg_local_t<uint32_t> target(*address);
  target.fetch_add(val);
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_local_ptr_pt<uint64_t>& address,
    uint64_t val) {
  dpcpp_atomic_ref_rlx_wg_local_t<uint64_t> target(*address);
  target.fetch_add(val);
}

template <typename T>
static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(&address->real_, val.real_);
  atomicAdd(&address->imag_, val.imag_);
}

// Atomic multiplication implementation.

SYCL_ATOMIC_INTEGER(Mul, Numerics<uint8_t>::mul(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Mul, Numerics<int8_t>::mul(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Mul, Numerics<int16_t>::mul(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Mul, Numerics<int32_t>::mul(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Mul, Numerics<int64_t>::mul(a, b), int64_t)

SYCL_ATOMIC_FP(Mul, Numerics<float>::mul(a, b), float)
SYCL_ATOMIC_FP(Mul, Numerics<double>::mul(a, b), double)
SYCL_ATOMIC_FP(Mul, Numerics<at::Half>::mul(a, b), at::Half)
SYCL_ATOMIC_FP(Mul, Numerics<at::BFloat16>::mul(a, b), at::BFloat16)

} // namespace AtenIpexTypeXPU
} // namespace at
