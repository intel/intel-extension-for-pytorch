#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void logical_not_kernel(TensorIterator& iter) {
  // NOTE: We should not dispatch on types which aren't in below
  // ALL_TYPES_AND... Therefore, we add the check here.
  IPEX_DISPATCH_ALL_TYPES_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_dpcpp", [&]() {});

  IPEX_DISPATCH_ALL_TYPES_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> bool { return !a; });
      });
}

} // namespace impl

Tensor& logical_not_out(Tensor& result, const Tensor& self) {
  TensorIterator iter = TensorIteratorConfig()
                            .check_all_same_dtype(false)
                            .set_check_mem_overlap(true)
                            .add_output(result)
                            .add_input(self)
                            .build();
  impl::logical_not_kernel(iter);
  return result;
}

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::AtenIpexTypeXPU::logical_not_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
