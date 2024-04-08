#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/OpMathType.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct leaky_relu_out_functor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t x) const {
    opmath_t x_ = static_cast<opmath_t>(x);
    x_ = (x_ >= opmath_t(0)) ? x_ : x_ * negval;
    return x_;
  }

  leaky_relu_out_functor(opmath_t negval) : negval(negval) {}

 private:
  opmath_t negval;
};

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
        using opmath_t = at::opmath_type<scalar_t>;
        auto negval = negative_slope.to<opmath_t>();
        leaky_relu_out_functor<scalar_t> f(negval);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct leaky_relu_backward_out_functor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t grad_output_, scalar_t x_) const {
    opmath_t grad_output = static_cast<opmath_t>(grad_output_);
    opmath_t x = static_cast<opmath_t>(x_);
    if (x > opmath_t(0))
      return grad_output;
    else
      return grad_output * negval;
  }

  leaky_relu_backward_out_functor(opmath_t negval) : negval(negval) {}

 private:
  opmath_t negval;
};

Tensor& leaky_relu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& negative_slope,
    bool self_is_result,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, self);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "LeakyReLU_backward",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negval = negative_slope.to<opmath_t>();
        leaky_relu_backward_out_functor<scalar_t> f(negval);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU

} // namespace at
