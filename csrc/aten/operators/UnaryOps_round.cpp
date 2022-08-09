#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/zmath.h"
#include "oneapi/dpl/cmath"
namespace dpl = oneapi::dpl;

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);

IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL(
    round_out,
    [](scalar_t a) -> scalar_t {
      return dpl::nearbyintf(static_cast<float>(a));
    },
    unary_op);

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(remainder_out, TensorRemainderOp);

Tensor remainder(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(fmod_out, TensorFmodOp);

Tensor fmod(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
