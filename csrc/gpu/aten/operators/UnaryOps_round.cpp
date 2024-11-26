#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#ifdef USE_OVERRIDE_OP
#include <utils/CustomOperatorRegistration.h>
#endif
#include <utils/DPCPP.h>

#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/zmath.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_ALL_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);

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

#ifdef USE_OVERRIDE_OP
namespace {

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("ceil.out", TORCH_FN((&at::AtenIpexTypeXPU::ceil_out)));
  m.impl("floor.out", TORCH_FN((&at::AtenIpexTypeXPU::floor_out)));
}

} // namespace
#endif
