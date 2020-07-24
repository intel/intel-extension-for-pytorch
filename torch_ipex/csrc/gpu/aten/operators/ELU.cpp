#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>
#include <core/DPCPP.h>
#include "Loops.h"

DPCPP_DEF_K1(SyclOpElu);
DPCPP_DEF_K1(SyclOpEluBackward);

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor& elu_out(
    Tensor& out,
    const Tensor& self,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(out);
  iter.add_input(self);
  iter.build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu",
      [&]() {
        auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
        auto poscoef = scale.to<scalar_t>();
        auto negiptocoef = input_scale.to<scalar_t>();

        dpcpp_kernel_for_tensor_iter<DPCPP_K(SyclOpElu)>(
            iter, [=](scalar_t x) -> scalar_t {
              x = x <= 0
                  ? (Numerics<scalar_t>::exp(x * negiptocoef) - 1) * negcoef
                  : x * poscoef;
              return x;
            });
      });

  return out;
}

Tensor elu(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
  Tensor result = at::empty(self.sizes(), self.options());
  at::AtenIpexTypeDPCPP::elu_out(result, self, alpha, scale, input_scale);
  return result;
}

Tensor& elu_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale,
    const Tensor& output) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(grad_output);
  iter.add_input(output);
  iter.build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "elu_backward", [&]() {
        auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
        auto poscoef = scale.to<scalar_t>();
        auto negiptocoef = input_scale.to<scalar_t>();

        dpcpp_kernel_for_tensor_iter<DPCPP_K(SyclOpEluBackward)>(
            iter, [=](scalar_t grad_output, scalar_t output) -> scalar_t {
              if (output <= 0)
                return grad_output * negiptocoef * (output + negcoef);
              else
                return grad_output * poscoef;
            });
      });
  return grad_input;
}

Tensor elu_backward(
    const Tensor& grad_output,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale,
    const Tensor& output) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeDPCPP::elu_backward_out(
      grad_input, grad_output, alpha, scale, input_scale, output);
}

Tensor& elu_(Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
  return at::AtenIpexTypeDPCPP::elu_out(self, self, alpha, scale, input_scale);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
