#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/AtenIpexTypeXPU.h>
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
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, iter.dtype(1), "logical_not_dpcpp", [&]() {
        using self_t = scalar_t;
        IPEX_DISPATCH_ALL_TYPES_AND2(
            kBool, kHalf, iter.dtype(0), "logical_not_dpcpp", [&]() {
              dpcpp_kernel_for_tensor_iter(iter, [](self_t a) -> scalar_t {
                return static_cast<scalar_t>(!a);
              });
            });
      });
}

} // namespace impl

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::AtenIpexTypeXPU::logical_not_out(self, self);
}

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

} // namespace AtenIpexTypeXPU
} // namespace at
