#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename accscalar_t>
struct hardtanh_out_functor {
  scalar_t operator()(scalar_t x_) const {
    auto x = static_cast<accscalar_t>(x_);
    if (at::_isnan(x)) {
      return x;
    }
    return (Numerics<accscalar_t>::min(
        Numerics<accscalar_t>::max(static_cast<accscalar_t>(x), min_), max_));
  }

  hardtanh_out_functor(accscalar_t min_, accscalar_t max_)
      : min_(min_), max_(max_) {}

 private:
  accscalar_t min_;
  accscalar_t max_;
};

Tensor& hardtanh_out(
    const Tensor& self,
    const Scalar& min_val,
    const Scalar& max_val,
    Tensor& out) {
  checkBackend("hardtanh", out, self.options().backend());
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_clip>(
      TensorIterator::unary_op,
      out,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "hardtanh",
            [&]() {
              using opmath_t = at::opmath_type<scalar_t>;
              auto min_ = min_val.to<opmath_t>();
              auto max_ = max_val.to<opmath_t>();
              hardtanh_out_functor<scalar_t, opmath_t> f(min_, max_);
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      },
      /* alpha = */ min_val.toFloat(),
      /* beta = */ max_val.toFloat());
}

Tensor hardtanh(
    const Tensor& self,
    const Scalar& min_val,
    const Scalar& max_val) {
  TORCH_CHECK(!self.is_sparse(), "hardtanh(dpcpp_sparse) is not supported.");
  Tensor result = at::empty_like(self);
  at::AtenIpexTypeXPU::hardtanh_out(self, min_val, max_val, result);
  return result;
}

Tensor& hardtanh_(Tensor& self, const Scalar& min_val, const Scalar& max_val) {
  return at::AtenIpexTypeXPU::hardtanh_out(self, min_val, max_val, self);
}

template <typename scalar_t, typename accscalar_t>
struct hardtanh_backward_out_functor {
  scalar_t operator()(scalar_t grad_output, scalar_t x) const {
    accscalar_t grad_output_ = static_cast<accscalar_t>(grad_output);
    accscalar_t x_ = static_cast<accscalar_t>(x);
    if (x_ <= min_ || x_ >= max_)
      return accscalar_t(0);
    else
      return grad_output_;
  }

  hardtanh_backward_out_functor(accscalar_t min_, accscalar_t max_)
      : min_(min_), max_(max_) {}

 private:
  accscalar_t min_;
  accscalar_t max_;
};

Tensor& hardtanh_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& min_val,
    const Scalar& max_val,
    Tensor& grad_input) {
  checkBackend(
      "hardtanh_backward", {grad_input, grad_output}, self.options().backend());
  // Compare the norm and maxnorm value.
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(grad_input)
                  .add_input(grad_output)
                  .add_input(self)
                  .build();

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardtanh_backward",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto min_ = min_val.to<opmath_t>();
        auto max_ = max_val.to<opmath_t>();
        hardtanh_backward_out_functor<scalar_t, opmath_t> f(min_, max_);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });

  return grad_input;
}

Tensor hardtanh_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& min_val,
    const Scalar& max_val) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeXPU::hardtanh_backward_out(
      grad_output, self, min_val, max_val, grad_input);
}
} // namespace AtenIpexTypeXPU
} // namespace at
