#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct sinc_kernel_xpu_functor {
  scalar_t operator()(scalar_t a) const {
    if (a == scalar_t(0)) {
      return scalar_t(1);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_t product = c10::detail::pi<opmath_t>() * opmath_t{a};
      return static_cast<scalar_t>(Numerics<opmath_t>::sin(product) / product);
    }
  }
};

void sinc_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "sinc",
      [&]() {
        sinc_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

Tensor& sinc_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  sinc_kernel_xpu(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
