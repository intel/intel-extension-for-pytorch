#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/Unary.h"
#include "comm/zmath.h"

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

void abs_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Bool,
      iter.common_dtype(),
      "abs",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::abs(a);
        });
      });
}

void angle_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "angle", [&]() {
    dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
      return at::AtenIpexTypeXPU::angle_impl(a);
    });
  });
}

void conj_physical_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.common_dtype(),
      "conj",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return at::AtenIpexTypeXPU::conj_impl(a);
        });
      });
}

} // namespace impl

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);

IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL(
    round_out,
    [](scalar_t a) -> scalar_t { return ::nearbyintf(static_cast<float>(a)); },
    unary_op);

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(remainder_out, TensorRemainderOp);

IPEX_UNARY_LOOPS_FUNC_ALL_ALL_COMPLEX(
    neg_out,
    Numerics<scalar_t>::neg,
    unary_op);

Tensor remainder(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(fmod_out, TensorFmodOp);

Tensor fmod(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

Tensor& conj_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  // IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(),
  // "conj_xpu", [&]() {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "conj_xpu", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t a) -> scalar_t { return conj_impl(a); });
      });

  return out;
}

Tensor& reciprocal_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "reciprocal_xpu",
      [&] {
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t a) -> scalar_t {
          return static_cast<scalar_t>(1.0) / a;
        });
      });
  return out;
}

Tensor& abs_out(const Tensor& self, Tensor& result) {
  return impl::unary_op_impl_with_complex_to_float_out(
      result, self, impl::abs_kernel, /*promotes_integer_to_float=*/false);
}

Tensor& conj_physical_out(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  impl::conj_physical_kernel(iter);
  return result;
}

Tensor& conj_physical_(Tensor& self) {
  if (!self.is_complex())
    return self;
  return at::AtenIpexTypeXPU::conj_physical_out(self, self);
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
