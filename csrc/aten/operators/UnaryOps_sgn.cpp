#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include "Loops.h"
#include "comm/zmath.h"

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(sgn, Numerics<scalar_t>::sgn, Real);

IPEX_OUT_ALL_CALLABLE_0_UNARY_OPS(sign_out, TensorSignOp);

Tensor sign(const Tensor& self) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::sign_out(out, self);
}

Tensor& sign_(Tensor& self) {
  return at::AtenIpexTypeXPU::sign_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
