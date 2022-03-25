#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "comm/zmath.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void mul_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "mul",
      [&]() {
        dpcpp_kernel_with_scalars(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; });
      });
}

static void div_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "div",
      [&]() {
        dpcpp_kernel_with_scalars(
            iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
      });
}

static void div_trunc_kernel_dpcpp(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div", [&] {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "div_trunc",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return trunc_impl(a / b);
              });
        });
  }
}

static void div_floor_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == kByte) {
    return div_trunc_kernel_dpcpp(iter);
  } else if (isIntegralType(iter.dtype(), false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div", [&] {
      dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        if ((a < 0) != (b < 0)) {
          const auto quot = a / b;
          const auto rem = a % b;
          return rem ? quot - 1 : quot;
        }

        return a / b;
      });
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "div_trunc_cpu",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                if (C10_UNLIKELY(b == 0)) {
                  // Divide by zero: return standard IEEE result
                  return a / b;
                }

                auto mod = std::fmod(a, b);
                auto div = (a - mod) / b;
                if ((mod != 0) && (b < 0) != (mod < 0)) {
                  div -= scalar_t(1);
                }

                scalar_t floordiv;
                if (div != 0) {
                  floordiv = std::floor(div);
                  if (div - floordiv > scalar_t(0.5)) {
                    floordiv += scalar_t(1.0);
                  }
                } else {
                  floordiv = DPCPP::copysign(0.0, scalar_cast<double>(a / b));
                }
                return floordiv;
              });
        });
  }
}

} // namespace impl

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::mul_kernel_dpcpp(iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::mul_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::mul_out(self, self, other);
}

Tensor mul(const Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::mul(self, wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::mul_(self, wrapped_scalar_tensor(other));
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  impl::div_kernel_dpcpp(iter);
  return result;
}

Tensor& div_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    auto iter = TensorIterator::binary_float_op(result, self, other);
    impl::div_kernel_dpcpp(iter);
  } else if (*rounding_mode == "trunc") {
    auto iter = TensorIterator::binary_op(result, self, other);
    impl::div_trunc_kernel_dpcpp(iter);
  } else if (*rounding_mode == "floor") {
    auto iter = TensorIterator::binary_op(result, self, other);
    impl::div_floor_kernel_dpcpp(iter);
  }
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  impl::div_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::div_out(self, self, other);
}

Tensor& floor_divide_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::div_trunc_kernel_dpcpp(iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::div_trunc_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::floor_divide_out(self, self, other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
