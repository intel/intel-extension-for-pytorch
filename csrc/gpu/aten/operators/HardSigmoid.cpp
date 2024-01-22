#include <ATen/ATen.h>
#include <core/Memory.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename accscalar_t>
struct hardsigmoid_out_functor {
  scalar_t operator()(scalar_t self_val) const {
    accscalar_t x = static_cast<accscalar_t>(self_val);
    return Numerics<accscalar_t>::min(
               Numerics<accscalar_t>::max(x + three, zero), six) *
        one_sixth;
  }

  hardsigmoid_out_functor(
      const accscalar_t zero,
      const accscalar_t one_sixth,
      const accscalar_t three,
      const accscalar_t six)
      : zero(zero), one_sixth(one_sixth), three(three), six(six) {}

 private:
  const accscalar_t zero;
  const accscalar_t one_sixth;
  const accscalar_t three;
  const accscalar_t six;
};

Tensor& hardsigmoid_out(const Tensor& self, Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_hardsigmoid>(
      TensorIterator::unary_float_op,
      out,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.common_dtype(),
            "hardsigmoid_out",
            [&]() {
              using accscalar_t = acc_type<scalar_t>;
              const accscalar_t zero(0.0f);
              const accscalar_t one_sixth(1.0f / 6.0f);
              const accscalar_t three(3.0f);
              const accscalar_t six(6.0f);
              hardsigmoid_out_functor<scalar_t, accscalar_t> f(
                  zero, one_sixth, three, six);
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      },
      /* alpha = */ 1.0f / 6.0f,
      /* beta = */ 1.0f / 2.0f);
}

template <typename scalar_t, typename accscalar_t>
struct hardsigmoid_backward_out_functor {
  scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    accscalar_t grad_val = static_cast<accscalar_t>(grad_val_);
    accscalar_t self_val = static_cast<accscalar_t>(self_val_);
    return (self_val > neg_three && self_val < three) ? grad_val * one_sixth
                                                      : zero;
  }

  hardsigmoid_backward_out_functor(
      const accscalar_t zero,
      const accscalar_t three,
      const accscalar_t neg_three,
      const accscalar_t one_sixth)
      : zero(zero), three(three), neg_three(neg_three), one_sixth(one_sixth) {}

 private:
  const accscalar_t zero;
  const accscalar_t three;
  const accscalar_t neg_three;
  const accscalar_t one_sixth;
};

Tensor& hardsigmoid_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    Tensor& grad_input) {
  TORCH_CHECK(self.numel() == grad_output.numel(), "different elements ...");
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(grad_input)
                  .add_input(grad_output)
                  .add_input(self)
                  .build();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "hardsigmoid_backward_out",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        const accscalar_t zero(0.0f);
        const accscalar_t three(3.0f);
        const accscalar_t neg_three(-3.0f);
        const accscalar_t one_sixth(1.0f / 6.0f);
        hardsigmoid_backward_out_functor<scalar_t, accscalar_t> f(
            zero, three, neg_three, one_sixth);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
