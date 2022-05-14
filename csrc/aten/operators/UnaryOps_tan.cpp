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
