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

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(expm1_out, Numerics<scalar_t>::expm1, Real);

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(exp, Numerics<scalar_t>::exp, Real);

} // namespace AtenIpexTypeXPU
} // namespace at
