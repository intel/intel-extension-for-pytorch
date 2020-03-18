#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>

#include "Loops.h"


using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

DPCPP_DEF_K1(softplus_forward);
DPCPP_DEF_K1(softplus_backward);

Tensor& softplus_out(Tensor &out, const Tensor &self, Scalar beta, Scalar threshold) {
  checkBackend("softplus_forward", {out}, self.type().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(out);
  iter.add_input(self);
  iter.build();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF (iter.dtype(), "softplus_forward", [&]() {
    auto b = beta.to<scalar_t> ();
    auto t = threshold.to<scalar_t> ();
    dpcpp_kernel_for_tensor_iter<DPCPP_K(softplus_forward)>(iter, [=](scalar_t a)-> scalar_t {
      return (a * b > t ? a : Numerics<scalar_t>::log1p(Numerics<scalar_t>::exp(a * b)) / b);
    });
  });

  return out;
}

Tensor softplus(const Tensor &self, Scalar beta, Scalar threshold) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::softplus_out(out, self, beta, threshold);
}

Tensor& softplus_backward_out(
          Tensor &grad_input,
          const Tensor &grad_output,
          const Tensor &self,
          Scalar beta,
          Scalar threshold,
          const Tensor &output) {
  checkBackend("softplus_backward", {grad_input, grad_output, output}, self.type().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(grad_output);
  iter.add_input(output);
  iter.build();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF (iter.dtype(), "softplus_backward", [&]() {
    auto b = beta.to<scalar_t> ();
    auto t = threshold.to<scalar_t> ();
    dpcpp_kernel_for_tensor_iter<DPCPP_K(softplus_backward)>(iter,
        [=](scalar_t grad_output_data, scalar_t output_data)-> scalar_t {
      scalar_t beta_out = b * output_data;
      scalar_t exp_bo = Numerics<scalar_t>::exp(beta_out);
      return beta_out > t ? grad_output_data : grad_output_data * (exp_bo - 1) / exp_bo;
    });
  });

  return grad_input;
}

Tensor softplus_backward(
         const Tensor &grad_output,
         const Tensor &self,
         Scalar beta,
         Scalar threshold,
         const Tensor &output) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeDPCPP::softplus_backward_out(grad_input, grad_output, self, beta, threshold, output);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
