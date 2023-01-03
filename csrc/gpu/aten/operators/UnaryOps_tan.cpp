#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
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

IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(
    tan_out,
    Numerics<scalar_t>::tan,
    unary_float_op);

Tensor& tanh_out(const Tensor& self, Tensor& result) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_tanh>(
      TensorIterator::unary_float_op,
      result,
      self,
      [](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.common_dtype(),
            "tanh",
            [&]() {
              dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
                using opmath_t = at::opmath_type<scalar_t>;
                return Numerics<opmath_t>::tanh(static_cast<opmath_t>(a));
              });
            });
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at
