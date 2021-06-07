#pragma once

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include "Numerics.h"


using namespace xpu::dpcpp;

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<float>& address,
    float val) {
  uint32_t* address_as_ull = (uint32_t*)address;
  uint32_t assumed = *address_as_ull;
  uint32_t newval;

  dpcpp_multi_ptr<uint32_t, dpcpp_global_space> address_multi_ptr(
      address_as_ull);
  DPCPP::atomic<uint32_t> address_var(address_multi_ptr);

  do {
    newval = __float_as_int(val + __int_as_float(assumed));
  } while (!address_var.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<double>& address,
    double val) {
  unsigned long long * address_as_ull = (unsigned long long *)address;
  unsigned long long assumed = *address_as_ull;
  unsigned long long newval;

  dpcpp_multi_ptr<unsigned long long, dpcpp_global_space> address_multi_ptr(
      address_as_ull);
  DPCPP::atomic<unsigned long long> address_var(address_multi_ptr);

  do {
    newval = __double_as_long_long(val + __long_long_as_double(assumed));
  } while (!address_var.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<at::Half>& address,
    at::Half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int assumed = *address_as_ui;
  unsigned int newval;

  dpcpp_multi_ptr<unsigned int, dpcpp_global_space> address_multi_ptr(
      address_as_ui);
  DPCPP::atomic<unsigned int> address_var(address_multi_ptr);

  do {
    newval = assumed;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
    hsum = Numerics<at::Half>::add(hsum, val);
    newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                              : (newval & 0xffff0000) | hsum.x;
  } while (!address_var.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<at::BFloat16>& address,
    at::BFloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int assumed = *address_as_ui;
  unsigned int newval;

  dpcpp_multi_ptr<unsigned int, dpcpp_global_space> address_multi_ptr(
      address_as_ui);
  DPCPP::atomic<unsigned int> address_var(address_multi_ptr);

  do {
    newval = assumed;
    at::BFloat16 hsum;
    hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
    hsum = Numerics<at::BFloat16>::add(hsum, val);
    newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                              : (newval & 0xffff0000) | hsum.x;
  } while (!address_var.compare_exchange_strong(assumed, newval));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int>& address,
    int val) {
  dpcpp_multi_ptr<int, dpcpp_global_space> address_multi_ptr(address);
  DPCPP::atomic<int> address_var(address_multi_ptr);
  address_var.fetch_add(val);
}

static inline DPCPP_DEVICE void atomicAdd(
        const dpcpp_global_ptr_pt<long>& address,
        int val) {
  dpcpp_multi_ptr<long, dpcpp_global_space> address_multi_ptr(address);
  DPCPP::atomic<long> address_var(address_multi_ptr);
  address_var.fetch_add(val);
}

// (TODO) add support for atomicAdd
// dpcpp_global_ptr_pt<uint8_t *>
// dpcpp_global_ptr_pt<int8_t>
// dpcpp_global_ptr_pt<bool *>
// dpcpp_global_ptr_pt<int64_t>
// dpcpp_global_ptr_pt<double *>
