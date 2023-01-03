#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/OpMathType.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& elu_out(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_elu>(
      TensorIterator::unary_op,
      out,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "elu",
            [&]() {
              using opmath_t = at::opmath_type<scalar_t>;
              auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
              auto poscoef = scale.to<opmath_t>();
              auto negiptcoef = input_scale.to<opmath_t>();
              dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t a) -> scalar_t {
                opmath_t aop = static_cast<opmath_t>(a);
                return aop > 0 ? aop * poscoef
                               : std::expm1(aop * negiptcoef) * negcoef;
              });
            });
      },
      /* alpha = */ alpha.to<float>(),
      /* beta = */ 0.0f,
      1.0 == scale.to<float>() && 1.0 == input_scale.to<float>());
}

Tensor elu(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  Tensor result = at::empty_like(self);
  at::AtenIpexTypeXPU::elu_out(self, alpha, scale, input_scale, result);
  return result;
}

Tensor& elu_backward_out(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result,
    Tensor& grad_input) {
  return unary_out_with_onednn_and_loops_bw<
      dnnl::algorithm::eltwise_elu_use_dst_for_bwd>(
      TensorIterator::binary_op,
      grad_input,
      grad_output,
      self_or_result,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "elu_backward",
            [&]() {
              using opmath_t = at::opmath_type<scalar_t>;
              auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
              auto poscoef = scale.to<opmath_t>();
              auto negiptcoef = input_scale.to<opmath_t>();
              dpcpp_kernel_for_tensor_iter(
                  iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                    opmath_t aop = static_cast<opmath_t>(a);
                    opmath_t bop = static_cast<opmath_t>(b);

                    if (is_result) {
                      return bop <= 0 ? aop * negiptcoef * (bop + negcoef)
                                      : aop * poscoef;
                    } else {
                      return bop <= 0 ? aop * negiptcoef * negcoef *
                              std::exp(bop * negiptcoef)
                                      : aop * poscoef;
                    }
                  });
            });
      },
      /*alpha =*/alpha.to<float>(),
      0.0f,
      !is_result && input_scale.to<float>() == 1 && scale.to<float>() == 1);
}

Tensor elu_backward(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result) {
  Tensor grad_input = at::empty_like(grad_output);
  return at::AtenIpexTypeXPU::elu_backward_out(
      grad_output,
      alpha,
      scale,
      input_scale,
      is_result,
      self_or_result,
      grad_input);
}

Tensor& elu_(
    Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  return at::AtenIpexTypeXPU::elu_out(self, alpha, scale, input_scale, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
