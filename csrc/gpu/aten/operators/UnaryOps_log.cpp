#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>

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

Tensor& log_out(const Tensor& self, Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_log>(
      TensorIterator::unary_float_op, out, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.common_dtype(),
            "log_out",
            [&]() {
              dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
                return Numerics<scalar_t>::log(a);
              });
            });
      });
}

IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    log10_out,
    Numerics<scalar_t>::log10,
    unary_float_op);
IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMMON(
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
