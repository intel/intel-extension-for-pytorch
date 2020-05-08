#include <ATen/ATen.h>
#include <ATen/Context.h>

#include <core/DPCPP.h>
#include "Loops.h"

namespace at {
namespace AtenIpexTypeDPCPP {

DPCPP_DEF_K1(SyclOpLeakyElu);
DPCPP_DEF_K1(SyclOpLeakyEluBackward);

Tensor& leaky_relu_out(Tensor& out, const Tensor& self, Scalar negative_slope) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(out);
  iter.add_input(self);
  iter.build();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "LeakyReLU", [&]() {
    auto negval = negative_slope.to<scalar_t>();
    dpcpp_kernel_for_tensor_iter<DPCPP_K(SyclOpLeakyElu)>(
        iter, [=](scalar_t x) -> scalar_t {
          x = (x >= 0) ? x : x * negval;
          return x;
        });
  });
  return out;
}

Tensor leaky_relu(const Tensor& self, Scalar negative_slope) {
  Tensor result = at::empty(self.sizes(), self.options());
  at::AtenIpexTypeDPCPP::leaky_relu_out(result, self, negative_slope);
  return result;
}

Tensor& leaky_relu_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar negative_slope) {
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(grad_output);
  iter.add_input(self);
  iter.build();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      iter.dtype(), "LeakyReLU_backward", [&]() {
        auto negval = negative_slope.to<scalar_t>();

        dpcpp_kernel_for_tensor_iter<DPCPP_K(SyclOpLeakyEluBackward)>(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              if (x > 0)
                return grad_output;
              else
                return grad_output * negval;
            });
      });
  return grad_input;
}

Tensor leaky_relu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar negative_slope) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeDPCPP::leaky_relu_backward_out(
      grad_input, grad_output, self, negative_slope);
}

Tensor& leaky_relu_(Tensor& self, Scalar negative_slope) {
  return at::AtenIpexTypeDPCPP::leaky_relu_out(self, self, negative_slope);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
