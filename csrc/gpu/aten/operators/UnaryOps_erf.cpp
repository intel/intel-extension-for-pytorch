#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct erf_out_functor {
  scalar_t operator()(scalar_t a) const {
    return Numerics<scalar_t>::erf(a);
  }
};

Tensor& erf_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "erf",
      [&]() {
        erf_out_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct erfc_out_functor {
  scalar_t operator()(scalar_t a) const {
    return Numerics<scalar_t>::erfc(a);
  }
};

Tensor& erfc_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "erfc",
      [&]() {
        erfc_out_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct erfinv_out_functor {
  scalar_t operator()(scalar_t a) const {
    scalar_t b;
    TensorErfinvOp<scalar_t>()(b, a);
    return b;
  }
};

Tensor& erfinv_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "erfinv",
      [&]() {
        erfinv_out_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
