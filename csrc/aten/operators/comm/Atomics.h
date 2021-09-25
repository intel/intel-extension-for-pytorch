#pragma once

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

} // namespace AtenIpexTypeXPU
} // namespace at

// (TODO) add support for atomicAdd
// dpcpp_global_ptr_pt<uint8_t *>
// dpcpp_global_ptr_pt<int8_t>
// dpcpp_global_ptr_pt<bool *>
// dpcpp_global_ptr_pt<int64_t>
// dpcpp_global_ptr_pt<double *>
