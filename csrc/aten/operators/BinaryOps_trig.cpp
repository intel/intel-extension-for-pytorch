#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& tanh_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(grad_input)
                  .add_input(grad_output)
                  .add_input(output)
                  .build();

  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "tanh_backward_out", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t output, scalar_t z) -> scalar_t {
              return output * (1. - z * z);
            });
      });

  return grad_input;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  auto grad_input = at::empty({0}, grad_output.options());
  return at::tanh_backward_out(grad_input, grad_output, output);
}

void atan2_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "atan2",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a, scalar_t b) -> scalar_t {
              return Numerics<scalar_t>::atan2(a, b);
            });
      });
}

Tensor& atan2_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  atan2_kernel(iter);
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  atan2_kernel(iter);
  return iter.output();
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::atan2_out(self, self, other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
