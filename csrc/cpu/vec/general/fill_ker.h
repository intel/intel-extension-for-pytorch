#pragma once

#include <ATen/cpu/vec/vec.h>
#include "utils/SysUtil.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
#if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
