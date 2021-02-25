#pragma once

#include <core/DPCPP.h>

namespace at {
namespace dpcpp {

template <typename reduce_op, typename nd_item_id, typename local_shared>
DPCPP_DEVICE static inline void simple_reduce(
  nd_item_id item_id,
  const local_shared& local_shared_mem,
  reduce_op bin_op) {
  auto local_idx = item_id.get_local_id(0);
  auto group_size = item_id.get_local_range().size();

  decltype(group_size) __k = 1;
  do {
    item_id.barrier(DPCPP::access::fence_space::local_space);
    if (local_idx % (2 * __k) == 0 && local_idx + __k < group_size) {
      local_shared_mem[local_idx] = bin_op(
        local_shared_mem[local_idx], local_shared_mem[local_idx + __k]);
    }
    __k *= 2;
  } while (__k < group_size);
  item_id.barrier(DPCPP::access::fence_space::local_space);
}

} // namespace dpcpp
} // namespace at
