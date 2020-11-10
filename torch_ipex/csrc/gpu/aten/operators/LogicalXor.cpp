#include <ATen/ATen.h>
#include <utils/ATDispatch.h>
#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

DPCPP_DEF_K1(logical_xor);
static void logical_xor_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, iter.dtype(1), "logical_xor_dpcpp", [&]() {
        using self_t = scalar_t;
        IPEX_DISPATCH_ALL_TYPES_AND2(
            kBool, kHalf, iter.dtype(2), "logical_xor_dpcpp", [&]() {
              using other_t = scalar_t;
              IPEX_DISPATCH_ALL_TYPES_AND2(
                  kBool, kHalf, iter.dtype(0), "logical_xor_dpcpp", [&]() {
                    dpcpp_kernel_for_tensor_iter<DPCPP_K(
                        logical_xor, self_t, other_t)>(
                        iter, [](self_t a, other_t b) -> scalar_t {
                          return static_cast<scalar_t>(bool(a) != bool(b));
                        });
                  });
            });
      });
}

} // namespace impl

Tensor& logical_xor_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto iter = TensorIteratorConfig()
  .check_all_same_dtype(false)
  .set_check_mem_overlap(true)
  .add_output(result)
  .add_input(self)
  .add_input(other)
  .build();
  impl::logical_xor_kernel(iter);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
