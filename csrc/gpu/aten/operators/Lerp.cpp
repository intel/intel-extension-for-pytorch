#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lerp_tensor_kernel",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(
            iter,
            [](scalar_t self_val,
               scalar_t end_val,
               scalar_t weight_val) -> scalar_t {
              opmath_t self_val_f = self_val;
              opmath_t end_val_f = end_val;
              opmath_t weight_val_f = weight_val;
              return (Numerics<scalar_t>::abs(weight_val_f) < 0.5f)
                  ? self_val_f + weight_val_f * (end_val_f - self_val_f)
                  : end_val_f -
                      (end_val_f - self_val_f) * (opmath_t{1} - weight_val_f);
            });
      });
}

void lerp_scalar_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lerp_tensor_kernel",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto weight_val = weight.to<opmath_t>();
        dpcpp_kernel_with_scalars(
            iter, [=](scalar_t self_val, scalar_t end_val) {
              opmath_t self_val_f = self_val;
              opmath_t end_val_f = end_val;
              return (Numerics<scalar_t>::abs(weight_val) < 0.5f)
                  ? self_val_f + weight_val * (end_val_f - self_val_f)
                  : end_val_f -
                      (end_val_f - self_val_f) * (opmath_t{1} - weight_val);
            });
      });
}

} // namespace impl

Tensor& lerp_out(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight,
    Tensor& out) {
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .add_input(end)
                  .add_input(weight)
                  .build();
  impl::lerp_tensor_kernel(iter);
  return out;
}

Tensor& lerp_out(
    const Tensor& self,
    const Tensor& end,
    const Scalar& weight,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, end);
  impl::lerp_scalar_kernel(iter, weight);
  return out;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Tensor& weight) {
  auto iter = TensorIteratorConfig()
                  .add_output(self)
                  .add_input(self)
                  .add_input(end)
                  .add_input(weight)
                  .build();
  impl::lerp_tensor_kernel(iter);
  return self;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Scalar& weight) {
  auto iter = TensorIterator::binary_op(self, self, end);
  impl::lerp_scalar_kernel(iter, weight);
  return self;
}

Tensor lerp(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(self, end, weight, result);
}

Tensor lerp(const Tensor& self, const Tensor& end, const Scalar& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(self, end, weight, result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
