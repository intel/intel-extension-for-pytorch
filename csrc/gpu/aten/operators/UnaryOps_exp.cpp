#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_UNARY_AND_ALL_OPS_COMMON(
    expm1_out,
    Numerics<scalar_t>::expm1,
    unary_float_op,
    FLOATING_TYPES)

Tensor& exp_out(const Tensor& self, Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_exp>(
      TensorIterator::unary_float_op, out, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.common_dtype(),
            "exp_out",
            [&]() {
              dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
                return Numerics<scalar_t>::exp(a);
              });
            });
      });
}

IPEX_UNARY_AND_ALL_OPS_COMMON(
    exp2_out,
    Numerics<scalar_t>::exp2,
    unary_float_op,
    FLOATING_TYPES)

} // namespace AtenIpexTypeXPU
} // namespace at
