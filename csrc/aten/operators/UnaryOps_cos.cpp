#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    cos_out,
    Numerics<scalar_t>::cos,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    acos_out,
    Numerics<scalar_t>::acos,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    cosh_out,
    Numerics<scalar_t>::cosh,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    acosh_out,
    Numerics<scalar_t>::acosh,
    unary_float_op);

} // namespace AtenIpexTypeXPU
} // namespace at
