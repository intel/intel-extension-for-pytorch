#include <ATen/ATen.h>
#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include <oneDNN/oneDNN.h>
#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erf, Numerics<scalar_t>::erf, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erfc, Numerics<scalar_t>::erfc, Real);

IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(erfinv_out, TensorErfinvOp);

Tensor& erfinv_(Tensor& self) {
  return at::AtenIpexTypeXPU::erfinv_out(self, self);
}

Tensor erfinv(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::erfinv_out(result, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
