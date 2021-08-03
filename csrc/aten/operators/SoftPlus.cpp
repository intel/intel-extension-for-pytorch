#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

DPCPP_DEF_K1(softplus_forward);
DPCPP_DEF_K1(softplus_backward);

Tensor& softplus_out(
    Tensor& out,
    const Tensor& self,
    Scalar beta,
    Scalar threshold) {
  checkBackend("softplus_forward", {out}, self.options().backend());
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "softplus_forward",
      [&]() {
        auto b = beta.to<scalar_t>();
        auto t = threshold.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(softplus_forward)>(
            iter, [=](scalar_t a) -> scalar_t {
              return (
                  a * b > t ? a
                            : Numerics<scalar_t>::log1p(
                                  Numerics<scalar_t>::exp(a * b)) /
                          b);
            });
      });

  return out;
}

Tensor softplus(const Tensor& self, Scalar beta, Scalar threshold) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::softplus_out(out, self, beta, threshold);
}

Tensor& softplus_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar beta,
    Scalar threshold,
    const Tensor& output) {
  checkBackend(
      "softplus_backward",
      {grad_input, grad_output, output},
      self.options().backend());
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(grad_input)
                  .add_input(grad_output)
                  .add_input(output)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "softplus_backward", [&]() {
        auto b = beta.to<scalar_t>();
        auto t = threshold.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(softplus_backward)>(
            iter,
            [=](scalar_t grad_output_data, scalar_t output_data) -> scalar_t {
              scalar_t beta_out = b * output_data;
              scalar_t exp_bo = Numerics<scalar_t>::exp(beta_out);
              return beta_out > t ? grad_output_data
                                  : grad_output_data * (exp_bo - 1) / exp_bo;
            });
      });

  return grad_input;
}

Tensor softplus_backward(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar beta,
    Scalar threshold,
    const Tensor& output) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeXPU::softplus_backward_out(
      grad_input, grad_output, self, beta, threshold, output);
}

} // namespace AtenIpexTypeXPU
} // namespace at
