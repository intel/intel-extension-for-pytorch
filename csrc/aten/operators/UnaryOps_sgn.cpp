#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

IPEX_UNARY_LOOPS_FUNC_ALL_ALL_COMPLEX(
    neg_out,
    Numerics<scalar_t>::neg,
    unary_op);

IPEX_UNARY_LOOPS_FUNC_COMPLEX(sgn_xpu, at::AtenIpexTypeXPU::sgn_impl, unary_op);

Tensor& sgn_out(const Tensor& self, Tensor& out) {
  if (self.is_complex()) {
    return at::AtenIpexTypeXPU::sgn_xpu(self, out);
  } else {
    return at::AtenIpexTypeXPU::sign_out(out, self);
  }
}

Tensor& sign_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter(iter, [](bool a) { return a; });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "sign_xpu", [&] {
          dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
            scalar_t zero = scalar_t(0);
            return (zero < a) - (a < zero);
          });
        });
  }
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
