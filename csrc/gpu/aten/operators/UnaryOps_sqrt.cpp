#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>
#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct sqrt_kernel_xpu_functor {
  scalar_t operator()(scalar_t a) const {
    return std::sqrt(a);
  }
};

void sqrt_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "sqrt",
      [&]() {
        sqrt_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

template <typename scalar_t>
struct sqrt_out_functor {
  scalar_t operator()(scalar_t a) const {
    return Numerics<scalar_t>::sqrt(a);
  }
};

Tensor& sqrt_out(const Tensor& self, Tensor& result) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_sqrt>(
      TensorIterator::unary_float_op,
      result,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.common_dtype(),
            "sqrt",
            [&]() {
              sqrt_out_functor<scalar_t> f;
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      });
}

template <typename scalar_t>
struct rsqrt_kernel_xpu_functor {
  scalar_t operator()(scalar_t a) const {
    return Numerics<scalar_t>::rsqrt(a);
  }
};

void rsqrt_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "rsqrt",
      [&]() {
        rsqrt_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

Tensor& rsqrt_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  rsqrt_kernel_xpu(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
