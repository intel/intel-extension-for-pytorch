#pragma once

namespace torch_ipex {
namespace cpu {
namespace kernel {

template <typename T>
inline __attribute__((always_inline)) void zero_ker(T* out, int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(out + i) = 0;
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
