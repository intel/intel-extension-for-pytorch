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

Tensor& hardsigmoid_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
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
        dpcpp_kernel_for_tensor_iter(
            iter, [zero, one_sixth, three, six](scalar_t self_val) -> scalar_t {
              accscalar_t x = static_cast<accscalar_t>(self_val);
              return Numerics<accscalar_t>::min(
                         Numerics<accscalar_t>::max(x + three, zero), six) *
                  one_sixth;
            });
      });
  return out;
}

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
        dpcpp_kernel_for_tensor_iter(
            iter,
            [zero, three, neg_three, one_sixth](
                scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
              accscalar_t grad_val = static_cast<accscalar_t>(grad_val_);
              accscalar_t self_val = static_cast<accscalar_t>(self_val_);
              return (self_val > neg_three && self_val < three)
                  ? grad_val * one_sixth
                  : zero;
            });
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
