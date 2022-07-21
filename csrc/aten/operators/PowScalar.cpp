#include <core/TensorImplUtils.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void sqrt_kernel_xpu(TensorIterator& iter);
void rsqrt_kernel_xpu(TensorIterator& iter);
void reciprocal_kernel_xpu(TensorIterator& iter);

template <typename scalar_t>
void pow_tensor_scalar_kernel_impl(TensorIterator& iter, const Scalar& exp) {
  const auto double_exp = exp.to<double>();
  // 0.5 (sqrt), -0.5 (rsqrt) and -1 (reciprocal) specializations are handled
  // in pow_tensor_scalar_kernel
  if (double_exp == 2) {
    dpcpp_kernel_for_tensor_iter(
        iter, [](scalar_t base) -> scalar_t { return base * base; });
  } else if (double_exp == 3) {
    dpcpp_kernel_for_tensor_iter(
        iter, [](scalar_t base) -> scalar_t { return base * base * base; });
  } else if (double_exp == -2) {
    dpcpp_kernel_for_tensor_iter(
        iter, [](scalar_t base) -> scalar_t { return 1.0 / (base * base); });
  } else {
    dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t base) -> scalar_t {
      return Numerics<scalar_t>::pow(base, double_exp);
    });
  }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, const Scalar& exp_scalar) {
  // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
  if (!exp_scalar.isComplex()) {
    if (exp_scalar.equal(0.5)) {
      return sqrt_kernel_xpu(iter);
    } else if (exp_scalar.equal(-0.5)) {
      return rsqrt_kernel_xpu(iter);
    } else if (exp_scalar.equal(-1.0)) {
      return reciprocal_kernel_xpu(iter);
    }
  }

  if (isComplexType(iter.common_dtype()) || exp_scalar.isComplex()) {
    IPEX_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "pow", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t base) -> scalar_t {
        return Numerics<scalar_t>::pow(base, exp);
      });
    });
  } else if (
      isFloatingType(iter.common_dtype()) || exp_scalar.isIntegral(false)) {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "pow_xpu", [&]() {
          // const auto exp = exp_scalar.to<scalar_t>();
          pow_tensor_scalar_kernel_impl<scalar_t>(iter, exp_scalar);
        });
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "invalid combination of type in Pow function, common dtype:",
        iter.common_dtype(),
        "exp is integral?",
        exp_scalar.isIntegral(false));
  }
}

Tensor& pow_out(const Tensor& base, const Scalar& exp, Tensor& result) {
  TORCH_CHECK(
      !(isIntegralType(base.scalar_type(), true) && exp.isIntegral(true) &&
        exp.toLong() < 0),
      "Intergers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  TORCH_CHECK(
      at::can_cast(common_dtype, result.scalar_type()),
      "result type ",
      common_dtype,
      "can't be cast to the desired output type ",
      result.scalar_type());

  if (exp.isComplex() && (exp.toComplexDouble() == 0.0)) {
    result.resize_as_(base).fill_(1);
  } else if (exp.isComplex() && (exp.toComplexDouble() == 1.0)) {
    result.resize_as_(base).fill_(base);
  } else if (!exp.isComplex() && (exp.toDouble() == 0.0)) {
    result.resize_as_(base).fill_(1);
  } else if (!exp.isComplex() && (exp.toDouble() == 1.0)) {
    result.resize_as_(base).copy_(base);
  } else {
    auto iter = TensorIterator::unary_op(result, base.to(common_dtype));
    pow_tensor_scalar_kernel(iter, exp);
  }
  return result;
}

Tensor pow(const Tensor& base, Scalar exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result =
      at::empty_like(base, base.options().dtype(dtype), MemoryFormat::Preserve);
  return at::AtenIpexTypeXPU::pow_out(base, exp, result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
