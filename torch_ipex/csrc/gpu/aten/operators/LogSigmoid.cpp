#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

DPCPP_DEF_K1(DPCPPPOpLogSigmoid);
std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out(
    Tensor& output,
    Tensor& buffer,
    const Tensor& self) {
  checkBackend("log_sigmoid_forward", output, self.options().backend());
  // Compare the norm and maxnorm value.
  auto iter = TensorIteratorConfig()
  .set_check_mem_overlap(true)
  .add_output(output)
  .add_input(self)
  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "log_sigmoid_forward",
      [&]() {
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPPOpLogSigmoid)>(
            iter, [=](scalar_t x) -> scalar_t {
              const scalar_t max = Numerics<scalar_t>::max(0, -x);
              const scalar_t z = Numerics<scalar_t>::exp(-max) +
                  Numerics<scalar_t>::exp(-x - max);
              return -(max + Numerics<scalar_t>::log(z));
            });
      });

  return std::tuple<Tensor&, Tensor&>{output, buffer};
}

std::tuple<Tensor, Tensor> log_sigmoid_forward(const Tensor& self) {
  TORCH_CHECK(
      !self.is_sparse(), "log_sigmoid_forward(dpcpp_sparse) is not supported.");
  Tensor buffer = at::empty({0}, self.options());
  Tensor result = at::empty(self.sizes(), self.options());
  at::AtenIpexTypeXPU::log_sigmoid_forward_out(result, buffer, self);
  return std::tuple<Tensor, Tensor>{result, buffer};
}

DPCPP_DEF_K1(DPCPPPOpLogSigmoidBackward);
Tensor& log_sigmoid_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  checkBackend(
      "log_sigmoid_backward",
      {grad_input, grad_output},
      self.options().backend());
  // Compare the norm and maxnorm value.
  auto iter = TensorIteratorConfig()
  .set_check_mem_overlap(true)
  .add_output(grad_input)
  .add_input(grad_output)
  .add_input(self)
  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "log_sigmoid_backward", [&]() {
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPPOpLogSigmoidBackward)>(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              const scalar_t max = Numerics<scalar_t>::max(0, -x);
              const scalar_t z = Numerics<scalar_t>::exp(-max) +
                  Numerics<scalar_t>::exp(-x - max);
              scalar_t max_deriv = 0.f;
              scalar_t sign = -1.f;
              if (x < 0.f) {
                max_deriv = -1.f;
                sign = 1.f;
              }
              return grad_output * (-max_deriv - sign * ((z - 1.f) / z));
            });
      });

  return grad_input;
}

Tensor log_sigmoid_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeXPU::log_sigmoid_backward_out(
      grad_input, grad_output, self, buffer);
}
} // namespace AtenIpexTypeXPU
} // namespace at
