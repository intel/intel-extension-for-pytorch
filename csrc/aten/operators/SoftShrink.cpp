#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void softshrink_forward(TensorIterator& iter, Scalar lambd) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "softshrink_forward",
      [&]() {
        auto lambd_data = lambd.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          if (x > lambd_data) {
            return (x - lambd_data);
          } else if (x < -lambd_data) {
            return (x + lambd_data);
          } else {
            return ScalarConvert<int, scalar_t>::to(0);
          }
        });
      });
}

static void softshrink_backward(TensorIterator& iter, Scalar lambd) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "softshrink_backward", [&]() {
        auto lambd_data = lambd.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t grad_output, scalar_t input) -> scalar_t {
              if (input > lambd_data || input < -lambd_data) {
                return grad_output;
              } else {
                return ScalarConvert<int, scalar_t>::to(0);
              }
            });
      });
}

} // namespace impl

Tensor& softshrink_out(Tensor& out, const Tensor& self, Scalar lambd) {
  checkBackend("softshrink_forward", {out}, self.options().backend());
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .build();
  impl::softshrink_forward(iter, lambd);
  return out;
}

Tensor softshrink(const Tensor& self, Scalar lambd) {
  TORCH_CHECK(
      lambd.to<double>() >= 0,
      "lambda must be greater or equal to 0, but found to be ",
      lambd.to<double>(),
      ".");
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::softshrink_out(out, self, lambd);
}

Tensor& softshrink_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  checkBackend(
      "softshrink_backward",
      {grad_input, grad_output},
      self.options().backend());
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(grad_input)
                  .add_input(grad_output)
                  .add_input(self)
                  .build();
  impl::softshrink_backward(iter, lambd);
  return grad_input;
}

Tensor softshrink_backward(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::softshrink_backward_out(grad_input, grad_output, self, lambd);
}
} // namespace AtenIpexTypeXPU
} // namespace at
