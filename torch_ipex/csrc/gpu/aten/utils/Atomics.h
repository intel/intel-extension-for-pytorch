#ifndef ATOMICS_INC
#define ATOMICS_INC

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>

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
    const dpcpp_global_ptr_pt<int>& address,
    int val) {
  dpcpp_multi_ptr<int, dpcpp_global_space> address_multi_ptr(address);
  DPCPP::atomic<int> address_var(address_multi_ptr);
  address_var.fetch_add(val);
}

#endif // ATOMICS_INC
