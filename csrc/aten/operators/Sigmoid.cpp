#include <ATen/ATen.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void sigmoid_backward(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& self) {
  gradInput.resize_as_(self);
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(gradInput)
                  .add_input(gradOutput)
                  .add_input(self)
                  .build();
  if (iter.dtype() == ScalarType::Half) {
    dpcpp_kernel_for_tensor_iter(
        iter, [=](at::Half go, at::Half in) -> at::Half {
          float in_float = (float)in;
          float go_float = (float)go;
          return (at::Half)(go * (1.f - in_float) * in_float);
        });
  } else {
    dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t go, scalar_t in) -> scalar_t {
          scalar_t one = (scalar_t)1.0;
          return go * (one - in) * in;
        });
  }
}

Tensor& sigmoid_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "_sigmoid_out",
      [&]() {
        out.resize_as_(self);
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t a) -> scalar_t {
          scalar_t one = (scalar_t)1.0;
          return one /
              (one + Numerics<scalar_t>::exp(-static_cast<scalar_t>(a)));
        });
      });

  return out;
}

Tensor& sigmoid_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      output.scalar_type(),
      "sigmoid_backward_out",
      [&]() {
        impl::sigmoid_backward<scalar_t>(grad_input, grad_output, output);
      });
  return grad_input;
}

} // namespace impl

Tensor& sigmoid_out(const Tensor& self, Tensor& out) {
  return impl::sigmoid_out(self, out);
}
Tensor& sigmoid_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  TORCH_CHECK(output.numel() == grad_output.numel(), "different elements ...");
  return impl::sigmoid_backward_out(grad_output, output, grad_input);
}

} // namespace AtenIpexTypeXPU
} // namespace at
