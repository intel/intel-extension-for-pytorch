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

IPEX_UNARY_LOOPS_FUNC_COMPLEX(sgn_xpu, at::AtenIpexTypeXPU::sgn_impl, unary_op);

IPEX_OUT_ALL_CALLABLE_0_UNARY_OPS(sign_out, TensorSignOp);

Tensor& sgn_out(const Tensor& self, Tensor& out) {
  if (self.is_complex()) {
    return at::AtenIpexTypeXPU::sgn_xpu(self, out);
  } else {
    return at::AtenIpexTypeXPU::sign_out(out, self);
  }
}

Tensor sign(const Tensor& self) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::sign_out(out, self);
}

Tensor& sign_(Tensor& self) {
  return at::AtenIpexTypeXPU::sign_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
