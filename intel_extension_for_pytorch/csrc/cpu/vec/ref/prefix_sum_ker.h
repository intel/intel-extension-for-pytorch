#pragma once

namespace torch_ipex {
namespace cpu {
namespace kernel {

template <typename T>
inline void prefix_sum(const T* src, T* dst, T init, int64_t n) {
  T sum = init;
  for (int64_t i = 0; i < n; i++) {
    sum += src[i];
    dst[i] = sum;
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
