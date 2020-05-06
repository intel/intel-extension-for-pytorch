#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>
#include <utils/Pairwise.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

IPEX_OUT_ALL_UNARY_FUNC_OPS(abs_out, Numerics<scalar_t>::abs, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(neg_out, Numerics<scalar_t>::neg, Real);

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sin_out, Numerics<scalar_t>::sin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(cosh_out, Numerics<scalar_t>::cosh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sinh_out, Numerics<scalar_t>::sinh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(acos_out, Numerics<scalar_t>::acos, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(asin_out, Numerics<scalar_t>::asin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(expm1_out, Numerics<scalar_t>::expm1, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(trunc_out, Numerics<scalar_t>::trunc, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(round_out, Numerics<scalar_t>::round, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log_out, Numerics<scalar_t>::log, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log10_out, Numerics<scalar_t>::log10, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log1p_out, Numerics<scalar_t>::log1p, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log2_out, Numerics<scalar_t>::log2, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(rsqrt_out, Numerics<scalar_t>::rsqrt, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sqrt_out, Numerics<scalar_t>::sqrt, Real);

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(cos, Numerics<scalar_t>::cos, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(tan, Numerics<scalar_t>::tan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(atan, Numerics<scalar_t>::atan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erf, Numerics<scalar_t>::erf, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erfc, Numerics<scalar_t>::erfc, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(exp, Numerics<scalar_t>::exp, Real);

} // namespace AtenIpexTypeDPCPP
} // namespace at
