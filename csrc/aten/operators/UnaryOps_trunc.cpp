#include <ATen/ATen.h>
#include <ATen/AtenIpexTypeXPU.h>
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

void frac_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(at::kHalf, iter.dtype(), "frac_xpu", [&]() {
    dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
      return a - Numerics<scalar_t>::trunc(a);
    });
  });
}
} // namespace impl

Tensor& frac_out(Tensor& result, const Tensor& self) {
  auto iter = TensorIterator::unary_op(result, self);
  impl::frac_kernel(iter);
  return result;
}

Tensor frac(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::frac_out(result, self);
}

Tensor& frac_(Tensor& self) {
  return at::AtenIpexTypeXPU::frac_out(self, self);
}

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(trunc_out, Numerics<scalar_t>::trunc, Real);

} // namespace AtenIpexTypeXPU
} // namespace at
