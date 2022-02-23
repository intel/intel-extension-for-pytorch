#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sin_out, Numerics<scalar_t>::sin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(cosh_out, Numerics<scalar_t>::cosh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sinh_out, Numerics<scalar_t>::sinh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(acos_out, Numerics<scalar_t>::acos, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(asin_out, Numerics<scalar_t>::asin, Real);

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(cos, Numerics<scalar_t>::cos, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(tan, Numerics<scalar_t>::tan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(atan, Numerics<scalar_t>::atan, Real);

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(tanh_out, Numerics<scalar_t>::tanh, Real);

Tensor& tanh_(Tensor& self) {
  return at::tanh_out(self, self);
}

Tensor tanh(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::tanh_out(result, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
