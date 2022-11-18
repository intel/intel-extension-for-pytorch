#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <utils/DPCPP.h>
#include "Loops.h"
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
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu",
      [&]() {
        auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
        auto poscoef = scale.to<scalar_t>();
        auto negiptocoef = input_scale.to<scalar_t>();

        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          x = x <= 0 ? (Numerics<scalar_t>::exp(x * negiptocoef) - 1) * negcoef
                     : x * poscoef;
          return x;
        });
      });

  return out;
}

Tensor elu(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  Tensor result = at::empty(self.sizes(), self.options());
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
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(grad_input)
                  .add_input(grad_output)
                  .add_input(self_or_result)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "elu_backward", [&]() {
        auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
        auto poscoef = scale.to<scalar_t>();
        auto negiptocoef = input_scale.to<scalar_t>();

        dpcpp_kernel_for_tensor_iter(
            iter,
            [=](scalar_t grad_output, scalar_t self_or_result) -> scalar_t {
              if (self_or_result <= 0) {
                if (is_result)
                  return grad_output * negiptocoef * (self_or_result + negcoef);
                else
                  return grad_output * negiptocoef * negcoef *
                      Numerics<scalar_t>::exp(self_or_result * negiptocoef);
              } else
                return grad_output * poscoef;
            });
      });
  return grad_input;
}

Tensor elu_backward(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result) {
  Tensor grad_input = at::empty({0}, grad_output.options());
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
