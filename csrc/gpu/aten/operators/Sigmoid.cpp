#include <ATen/ATen.h>
#include <core/Memory.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct sigmoid_out_functor {
  scalar_t operator()(scalar_t a) const {
    scalar_t one = (scalar_t)1.0;
    return one / (one + Numerics<scalar_t>::exp(-static_cast<scalar_t>(a)));
  }
};

Tensor& sigmoid_out(const Tensor& self, Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_logistic>(
      TensorIterator::unary_float_op, out, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.common_dtype(),
            "_sigmoid_out",
            [&]() {
              sigmoid_out_functor<scalar_t> f;
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      });
}

template <typename scalar_t>
struct sigmoid_backward_out_functor {
  at::Half operator()(at::Half go, at::Half in) const {
    float in_float = (float)in;
    float go_float = (float)go;
    return (at::Half)(go * (1.f - in_float) * in_float);
  }
};

template <typename scalar_t>
struct sigmoid_backward_out_functor_2 {
  scalar_t operator()(scalar_t go, scalar_t in) const {
    scalar_t one = (scalar_t)1.0;
    return go * (one - in) * in;
  }
};

Tensor& sigmoid_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  TORCH_CHECK(output.numel() == grad_output.numel(), "different elements ...");
  return unary_out_with_onednn_and_loops_bw<
      dnnl::algorithm::eltwise_logistic_use_dst_for_bwd>(
      TensorIterator::binary_op,
      grad_input,
      grad_output,
      output,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "sigmoid_backward_out",
            [&]() {
              if (iter.dtype() == ScalarType::Half) {
                sigmoid_backward_out_functor<scalar_t> f;
                dpcpp_kernel_for_tensor_iter(iter, f);
              } else {
                sigmoid_backward_out_functor_2<scalar_t> f;
                dpcpp_kernel_for_tensor_iter(iter, f);
              }
            });
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at
