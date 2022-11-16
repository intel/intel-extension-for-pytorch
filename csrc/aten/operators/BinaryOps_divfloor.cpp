#include <ATen/Context.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "comm/Numerics.h"
#include "comm/zmath.h"

#include "oneapi/dpl/cmath"
namespace dpl = oneapi::dpl;

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void div_trunc_kernel(TensorIterator& iter);

namespace impl {

template <typename scalar_t>
struct DivFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template <typename T>
struct MulFunctor {
  inline T operator()(T a, T b) const {
    return a * b;
  }
};

template <>
struct MulFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a && b;
  }
};

const char div_name[] = "div_kernel";
void div_true_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (iter.common_dtype() == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
    return;
  }
  if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_dpcpp", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto inv_b = opmath_t(1.0) / iter.scalar_value<opmath_t>(2);
          iter.remove_operand(2);
          dpcpp_kernel_for_tensor_iter(
              iter,
              BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                  MulFunctor<opmath_t>(), inv_b));
        });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_dpcpp", [&]() {
          DivFunctor<scalar_t> f;
          dpcpp_kernel_with_scalars(iter, f);
        });
  }
}

static void div_floor_kernel_dpcpp(TensorIterator& iter) {
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    // In the special case of unsigned integer division, floor division is
    // equivalent to truncation division (since the signs of the divisor and
    // dividend are always the same)
    return at::AtenIpexTypeXPU::div_trunc_kernel(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_dpcpp", [&]() {
      dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        if (c10::signs_differ(a, b)) {
          // Subtracts one from the results of truncation division if the
          // divisor and dividend have different sign(bit)s and the
          // remainder of the division is nonzero
          const auto quot = a / b;
          const auto rem = a % b;
          return rem ? quot - 1 : quot;
        }

        return a / b;
      });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        dtype,
        "div_floor_dpcpp",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          auto b = iter.scalar_value<accscalar_t>(2);
          if (DPCPP_UNLIKELY(b == 0)) {
            return div_true_kernel(iter);
          }

          auto inv_b = accscalar_t(1.0) / b;
          iter.remove_operand(2);
          dpcpp_kernel_for_tensor_iter(
              iter, [b, inv_b](scalar_t a) -> scalar_t {
                auto mod = Numerics<scalar_t>::fmod(a, b);
                auto div = (a - mod) * inv_b;
                if ((mod != 0) && (b < 0) != (mod < 0)) {
                  div -= scalar_t(1);
                }

                scalar_t floordiv;
                if (div != 0) {
                  floordiv = Numerics<scalar_t>::floor(div);
                  if (div - floordiv > scalar_t(0.5)) {
                    floordiv += scalar_t(1.0);
                  }
                } else {
                  floordiv =
                      Numerics<scalar_t>::copysign(scalar_t(0), a * inv_b);
                }
                return floordiv;
              });
        });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        dtype,
        "div_floor_dpcpp",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                if (DPCPP_UNLIKELY(b == 0)) {
                  return a / b;
                }

                auto mod = Numerics<scalar_t>::fmod(a, b);
                auto div = (a - mod) / b;
                if ((mod != 0) && (b < 0) != (mod < 0)) {
                  div -= scalar_t(1);
                }

                scalar_t floordiv;
                if (div != 0) {
                  floordiv = Numerics<scalar_t>::floor(div);
                  if (div - floordiv > scalar_t(0.5)) {
                    floordiv += scalar_t(1.0);
                  }
                } else {
                  floordiv = Numerics<scalar_t>::copysign(scalar_t(0), a / b);
                }
                return floordiv;
              });
        });
  }
}

} // namespace impl

Tensor& floor_divide_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::div_floor_kernel_dpcpp(iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::div_floor_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::floor_divide_out(self, other, self);
}

void div_floor_kernel(TensorIterator& iter) {
  impl::div_floor_kernel_dpcpp(iter);
}

} // namespace AtenIpexTypeXPU
} // namespace at
