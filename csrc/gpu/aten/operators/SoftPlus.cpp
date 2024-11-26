#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename accscalar_t>
struct softplus_out_functor {
  scalar_t operator()(scalar_t a_) const {
    accscalar_t a = static_cast<accscalar_t>(a_);
    return scalar_t(
        a * b > t
            ? a
            : Numerics<accscalar_t>::log1p(Numerics<accscalar_t>::exp(a * b)) /
                b);
  }

  softplus_out_functor(accscalar_t b, accscalar_t t) : b(b), t(t) {}

 private:
  accscalar_t b;
  accscalar_t t;
};

Tensor& softplus_out(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "softplus_forward",
      [&]() {
        using accscalar_t = at::opmath_type<scalar_t>;
        auto b = beta.to<accscalar_t>();
        auto t = threshold.to<accscalar_t>();
        softplus_out_functor<scalar_t, accscalar_t> f(b, t);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });

  return out;
}

Tensor softplus(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::softplus_out(self, beta, threshold, out);
}

} // namespace AtenIpexTypeXPU
} // namespace at
