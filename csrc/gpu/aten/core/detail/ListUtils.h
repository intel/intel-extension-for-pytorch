#pragma once

#include <ATen/ATen.h>
#include <limits>

namespace xpu {
namespace dpcpp {
namespace detail {

static inline int64_t sum_intlist(at::ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 0ll);
}

static inline int64_t prod_intlist(at::ArrayRef<int64_t> list) {
  return std::accumulate(
      list.begin(), list.end(), 1ll, std::multiplies<int64_t>());
}

} // namespace detail
} // namespace dpcpp
} // namespace xpu
