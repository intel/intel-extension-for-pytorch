#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(remainder_out, TensorCRemainderOp)

Tensor remainder(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(fmod_out, TensorCFmodOp)

Tensor fmod(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

IPEX_BINARY_LOOPS_FUNC_FLOAT_ALL_SCALAR(
    copysign_out,
    Numerics<scalar_t>::copysign,
    binary_float_op)

} // namespace AtenIpexTypeXPU
} // namespace at
