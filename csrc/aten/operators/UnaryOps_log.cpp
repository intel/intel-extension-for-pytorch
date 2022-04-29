#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/Unary.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    log_out,
    Numerics<scalar_t>::log,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    log10_out,
    Numerics<scalar_t>::log10,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL(
    log1p_out,
    Numerics<scalar_t>::log1p,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    log2_out,
    Numerics<scalar_t>::log2,
    unary_float_op);

Tensor& log1p_(Tensor& self) {
  at::log1p_out(self, self);
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
