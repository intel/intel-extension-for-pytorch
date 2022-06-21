#include <ATen/ATen.h>
#include <ATen/Context.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor& leaky_relu_out(
    const Tensor& self,
    const Scalar& negative_slope,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "LeakyReLU",
      [&]() {
        auto negval = negative_slope.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          x = (x >= 0) ? x : x * negval;
          return x;
        });
      });
  return out;
}

Tensor& leaky_relu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& negative_slope,
    bool self_is_result,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, self);

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "LeakyReLU_backward", [&]() {
        auto negval = negative_slope.to<scalar_t>();

        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              if (x > 0)
                return grad_output;
              else
                return grad_output * negval;
            });
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor& q_leaky_relu(Tensor& out, const Tensor& self, Scalar negative_slope) {
  float alpha = negative_slope.to<float>();
  xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_relu>(out, self, alpha, 0.0f);
  return out;
}

Tensor& leaky_relu_(Tensor& self, const Scalar& negative_slope) {
  return q_leaky_relu(self, self, negative_slope);
}

Tensor leaky_relu(const Tensor& self, const Scalar& negative_slope) {
  Tensor out = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      self.q_scale(),
      self.q_zero_point());
  auto result = q_leaky_relu(out, self, negative_slope);
  return result;
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
