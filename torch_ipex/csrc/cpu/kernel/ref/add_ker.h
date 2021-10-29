#pragma once

namespace torch_ipex {
namespace cpu {
namespace kernel {
namespace ref {

template <typename dst_type, typename src_type>
inline void add_ker(dst_type* inout, src_type* in, int len) {
  for (int i = 0; i < len; i++) {
    *(inout + i) += *(in + i);
  }
}

} // namespace ref
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
