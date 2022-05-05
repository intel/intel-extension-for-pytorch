#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_UNARY_AND_ALL_OPS(
    expm1_out,
    Numerics<scalar_t>::expm1,
    unary_float_op,
    FLOATING_TYPES)

IPEX_UNARY_AND_ALL_OPS(
    exp_out,
    Numerics<scalar_t>::exp,
    unary_float_op,
    FLOATING_AND_COMPLEX_TYPES)

IPEX_UNARY_AND_ALL_OPS(
    exp2_out,
    Numerics<scalar_t>::exp2,
    unary_float_op,
    FLOATING_TYPES)

} // namespace AtenIpexTypeXPU
} // namespace at
