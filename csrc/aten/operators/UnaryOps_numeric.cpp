#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename func_t>
static inline Tensor& unary_op_impl_with_complex_to_float_out(
    Tensor& result,
    const Tensor& self,
    const func_t& fn,
    bool promotes_integer_to_float) {
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toValueType(self.scalar_type());
    TORCH_CHECK(
        canCast(float_type, result.scalar_type()),
        "result type ",
        float_type,
        " can't be cast to the desired output type ",
        result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    Tensor complex_result = at::empty({0}, self.options());
    auto iter = TensorIterator::unary_op(complex_result, self);
    fn(iter);

    // Copies the complex result to the actual result and returns it
    at::native::resize_output(result, complex_result.sizes());
    result.copy_(at::real(complex_result));
    return result;
  }

  if (promotes_integer_to_float) {
    auto iter = TensorIterator::unary_float_op(result, self);
    fn(iter);
    iter.cast_outputs();
    return result;
  }

  auto iter = TensorIterator::unary_op(result, self);
  fn(iter);
  return result;
}

template <typename T>
static T abs_impl(T v) {
  return Numerics<T>::abs(v);
}

void abs_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Bool,
      iter.common_dtype(),
      "abs",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return abs_impl<scalar_t>(a); });
      });
}

void angle_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "angle", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return at::AtenIpexTypeXPU::angle_impl(a);
        });
      });
}

} // namespace impl

Tensor& abs_out(const Tensor& self, Tensor& result) {
  return impl::unary_op_impl_with_complex_to_float_out(
      result, self, impl::abs_kernel, /*promotes_integer_to_float=*/false);
}

Tensor& angle_out(const Tensor& self, Tensor& result) {
  return impl::unary_op_impl_with_complex_to_float_out(
      result, self, impl::angle_kernel, /*promotes_integer_to_float=*/true);
}

Tensor angle(const Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return at::angle_out(result, self);
  }
  Tensor result;
  auto iter = TensorIterator::unary_float_op(result, self);
  impl::angle_kernel(iter);
  return iter.output();
}

} // namespace AtenIpexTypeXPU
} // namespace at
