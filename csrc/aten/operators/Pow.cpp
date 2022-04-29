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
namespace impl {

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar);

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  // TODO: support complex dtype for power
  // if (isFloatingType(iter.dtype()) || isComplexType(iter.dtype())) {
  //   IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "pow", [&]() {
  //     dpcpp_kernel_for_tensor_iter(
  //         iter, [](scalar_t base, scalar_t exp) -> scalar_t {
  //           return Numerics<scalar_t>::pow(base, exp);
  //         });
  //   });

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "pow_xpu", [&]() {
        if (iter.is_cpu_scalar(1)) {
          const auto base = iter.scalar_value<scalar_t>(1);
          iter.remove_operand(1);
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t exp) -> scalar_t {
            return Numerics<scalar_t>::pow(base, exp);
          });
        } else if (iter.is_cpu_scalar(2)) {
          const auto exp = iter.scalar_value<scalar_t>(2);
          iter.remove_operand(2);
          pow_tensor_scalar_kernel(iter, exp);
        } else {
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t base, scalar_t exp) -> scalar_t {
                return Numerics<scalar_t>::pow(base, exp);
              });
        }
      });
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  if (isFloatingType(iter.dtype())) {
    const auto exp = exp_scalar.to<double>();
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "pow",
        [&]() {
          if (exp == 0.5) {
            dpcpp_kernel_for_tensor_iter(iter, [](scalar_t base) -> scalar_t {
              return Numerics<scalar_t>::sqrt(base);
            });
          } else if (exp == 2) {
            dpcpp_kernel_for_tensor_iter(
                iter, [](scalar_t base) -> scalar_t { return base * base; });
          } else if (exp == 3) {
            dpcpp_kernel_for_tensor_iter(iter, [](scalar_t base) -> scalar_t {
              return base * base * base;
            });
          } else if (exp == -0.5) {
            dpcpp_kernel_for_tensor_iter(iter, [](scalar_t base) -> scalar_t {
              return Numerics<scalar_t>::rsqrt(base);
            });
          } else if (exp == -1) {
            dpcpp_kernel_for_tensor_iter(
                iter, [](scalar_t base) -> scalar_t { return 1.0 / base; });
          } else if (exp == -2) {
            dpcpp_kernel_for_tensor_iter(iter, [](scalar_t base) -> scalar_t {
              return 1.0 / (base * base);
            });
          } else {
            dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t base) -> scalar_t {
              return Numerics<scalar_t>::pow(base, exp);
            });
          }
        });
  } else if (isComplexType(iter.dtype()) || exp_scalar.isComplex()) {
    IPEX_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "pow", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t base) -> scalar_t {
        return Numerics<scalar_t>::pow(base, exp);
      });
    });
  } else {
    const auto exp = exp_scalar.to<long>();
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "pow", [&]() {
      if (exp == 2) {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t base) -> scalar_t { return base * base; });
      } else if (exp == 3) {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t base) -> scalar_t { return base * base * base; });
      } else if (exp == -1) {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t base) -> scalar_t { return 1.0 / base; });
      } else if (exp == -2) {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t base) -> scalar_t {
          return 1.0 / (base * base);
        });
      } else {
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t base) -> scalar_t {
          return Numerics<scalar_t>::pow(base, exp);
        });
      }
    });
  }
}

} // namespace impl

Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto iter = TensorIterator::binary_op(result, base, exp);
  impl::pow_tensor_tensor_kernel(iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Tensor& base, const Scalar& exp) {
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
    impl::pow_tensor_scalar_kernel(iter, exp);
  }
  return result;
}

Tensor& pow_out(Tensor& result, const Scalar& base, const Tensor& exp) {
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else {
    at::AtenIpexTypeXPU::pow_out(result, wrapped_scalar_tensor(base), exp);
  }
  return result;
}

Tensor& pow_(Tensor& base, const Tensor& other) {
  return at::AtenIpexTypeXPU::pow_out(base, base, other);
}

Tensor& pow_(Tensor& base, Scalar alpha) {
  return at::AtenIpexTypeXPU::pow_out(base, base, alpha);
}

Tensor pow(const Tensor& base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty({0}, base.options().dtype(dtype));
  return at::AtenIpexTypeXPU::pow_out(result, base, exp);
}

Tensor pow(const Tensor& base, Scalar exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result =
      at::empty_like(base, base.options().dtype(dtype), MemoryFormat::Preserve);
  return at::AtenIpexTypeXPU::pow_out(result, base, exp);
}

Tensor pow(Scalar base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result =
      at::empty_like(exp, exp.options().dtype(dtype), MemoryFormat::Preserve);
  return at::AtenIpexTypeXPU::pow_out(result, base, exp);
}

} // namespace AtenIpexTypeXPU
} // namespace at
