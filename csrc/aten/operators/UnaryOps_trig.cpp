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
    sin_out,
    Numerics<scalar_t>::sin,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    asin_out,
    Numerics<scalar_t>::asin,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    sinh_out,
    Numerics<scalar_t>::sinh,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    asinh_out,
    Numerics<scalar_t>::asinh,
    unary_float_op);
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
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    tan_out,
    Numerics<scalar_t>::tan,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    atan_out,
    Numerics<scalar_t>::atan,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    tanh_out,
    Numerics<scalar_t>::tanh,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    atanh_out,
    Numerics<scalar_t>::atanh,
    unary_float_op);

} // namespace AtenIpexTypeXPU
} // namespace at
