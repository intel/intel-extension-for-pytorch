#include <ATen/ATen.h>

#include "Loops.h"


using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DP_DEF_K1(logical_xor);
static void logical_xor_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_xor_dpcpp", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(2), "logical_xor_dpcpp", [&]() {
      using other_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_xor_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter<DP_K(logical_xor, self_t, other_t)>(iter, [](self_t a, other_t b) -> scalar_t {
          return static_cast<scalar_t>(bool(a) != bool(b));
        });
      });
    });
  });
}

} // namespace impl

Tensor & logical_xor_out(Tensor & result, const Tensor & self, const Tensor & other) {
  TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.add_input(other);
  iter.build();
  impl::logical_xor_kernel(iter);
  return result;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
