#ifndef ATOMICS_INC
#define ATOMICS_INC

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <utils/Numerics.h>

using namespace at::dpcpp;

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
    const dpcpp_global_ptr_pt<at::Half>& address,
    at::Half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  dpcpp_multi_ptr<unsigned int, dpcpp_global_space> address_multi_ptr(
      address_as_ui);
  DPCPP::atomic<unsigned int> address_var(address_multi_ptr);

  do {
    assumed = old;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = Numerics<at::Half>::add(hsum, val);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
  } while (!address_var.compare_exchange_strong(old, assumed));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<at::BFloat16>& address,
    at::BFloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  dpcpp_multi_ptr<unsigned int, dpcpp_global_space> address_multi_ptr(
      address_as_ui);
  DPCPP::atomic<unsigned int> address_var(address_multi_ptr);

  do {
    assumed = old;
    at::BFloat16 hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = Numerics<at::BFloat16>::add(hsum, val);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
  } while (!address_var.compare_exchange_strong(old, assumed));
}

static inline DPCPP_DEVICE void atomicAdd(
    const dpcpp_global_ptr_pt<int>& address,
    int val) {
  dpcpp_multi_ptr<int, dpcpp_global_space> address_multi_ptr(address);
  DPCPP::atomic<int> address_var(address_multi_ptr);
  address_var.fetch_add(val);
}

// (TODO) add support for atomicAdd
// dpcpp_global_ptr_pt<uint8_t *>
// dpcpp_global_ptr_pt<int8_t>
// dpcpp_global_ptr_pt<bool *>
// dpcpp_global_ptr_pt<int64_t>
// dpcpp_global_ptr_pt<double *>

#endif // ATOMICS_INC
