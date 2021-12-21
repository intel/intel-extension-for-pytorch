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

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log_out, Numerics<scalar_t>::log, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log10_out, Numerics<scalar_t>::log10, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log1p_out, Numerics<scalar_t>::log1p, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log2_out, Numerics<scalar_t>::log2, Real);

Tensor& log1p_(Tensor& self) {
  at::AtenIpexTypeXPU::log1p_out(self, self);
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
