#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct hardtanh_out_functor {
  scalar_t operator()(scalar_t x) const {
    if (x < min_)
      return min_;
    else if (x > max_)
      return max_;
    else
      return x;
  }

  hardtanh_out_functor(scalar_t min_, scalar_t max_) : min_(min_), max_(max_) {}

 private:
  scalar_t min_;
  scalar_t max_;
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
              scalar_t min_ = min_val.to<scalar_t>();
              scalar_t max_ = max_val.to<scalar_t>();
              hardtanh_out_functor<scalar_t> f(min_, max_);
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

template <typename scalar_t>
struct hardtanh_backward_out_functor {
  scalar_t operator()(scalar_t grad_output, scalar_t x) const {
    if (x <= min_ || x >= max_)
      return 0;
    else
      return grad_output;
  }

  hardtanh_backward_out_functor(scalar_t min_, scalar_t max_)
      : min_(min_), max_(max_) {}

 private:
  scalar_t min_;
  scalar_t max_;
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
        auto min_ = min_val.to<scalar_t>();
        auto max_ = max_val.to<scalar_t>();
        hardtanh_backward_out_functor<scalar_t> f(min_, max_);
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
