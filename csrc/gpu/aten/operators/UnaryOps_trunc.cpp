#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct frac_kernel_functor {
  scalar_t operator()(scalar_t a) const {
    return a - Numerics<scalar_t>::trunc(a);
  }
};

void frac_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "frac_xpu", [&]() {
        frac_kernel_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

} // namespace impl

Tensor& frac_out(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  impl::frac_kernel(iter);
  return result;
}

template <typename scalar_t>
struct trunc_out_functor {
  scalar_t operator()(scalar_t a) const {
    return Numerics<scalar_t>::trunc(a);
  }
};

Tensor& trunc_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "trunc_out",
      [&]() {
        trunc_out_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
