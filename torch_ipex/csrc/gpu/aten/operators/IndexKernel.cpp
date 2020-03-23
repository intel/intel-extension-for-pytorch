#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Indexing.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Atomics.h>

#include "Loops.h"

DPCPP_DEF_K1(index_kernel);
DPCPP_DEF_K1(index_put_kernel);

namespace at {
namespace dpcpp {

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

static void index_kernel_dpcpp(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "index", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    dpcpp_index_kernel<DPCPP_K(index_kernel, scalar_t)>(
        iter,
        index_size,
        index_stride,
        // This lambda function only works in dpcpp kernel.
        [](dpcpp_global_ptr_pt<char> out_data,
           dpcpp_global_ptr_pt<char> in_data,
           int64_t offset) {
          *(dtype*)out_data = *(dtype*)(in_data + offset);
        });
  });
}
}
} // namespace at::dpcpp
