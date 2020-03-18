#ifndef ATOMICS_INC
#define ATOMICS_INC

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>


using namespace at::dpcpp;

static inline DP_DEVICE void atomicAdd(const dp_global_ptr_pt<float> &address, float val) {
  uint32_t* address_as_ull = (uint32_t*)address;
  uint32_t assumed = *address_as_ull;
  uint32_t newval;

  dp_multi_ptr<uint32_t, dp_global_space> address_multi_ptr(address_as_ull);
  DP::atomic<uint32_t> address_var(address_multi_ptr);

  do {
    newval = __float_as_int(val + __int_as_float(assumed));
  } while (!address_var.compare_exchange_strong(assumed, newval));
}

static inline DP_DEVICE void atomicAdd(const dp_global_ptr_pt<int> &address, int val) {
  dp_multi_ptr<int, dp_global_space> address_multi_ptr(address);
  DP::atomic<int> address_var(address_multi_ptr);
  address_var.fetch_add(val);
}


#endif // ATOMICS_INC
