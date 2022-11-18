#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Math.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

#include <ATen/Context.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& i0_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "i0_out",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i0(x));
        });
      });
  return out;
}

Tensor& special_ndtri_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "special_ndtri_out",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return calc_ndtri(a); });
      });
  return out;
}

Tensor& special_i0e_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "i0e",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i0e(x));
        });
      });
  return out;
}

Tensor& special_i1_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "i1", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i1(x));
        });
      });
  return out;
}

Tensor& special_i1e_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "i1e", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i1e(x));
        });
      });
  return out;
}

Tensor& special_entr_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "entr",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          if (at::_isnan(x)) {
            return x;
          } else if (x > 0) {
            return -x * Numerics<scalar_t>::log(x);
          } else if (x == 0) {
            return 0;
          }
          return Numerics<scalar_t>::lower_bound();
        });
      });
  return out;
}

Tensor& special_erfcx_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "erfcx", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return calc_erfcx(a); });
      });
  return out;
}

Tensor& xlogy_out(const Tensor& self, const Tensor& other, at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "xlogy",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t y) -> scalar_t {
          if (at::_isnan(y)) {
            return NAN;
          }
          if (x == 0) {
            return 0;
          }
          return x * Numerics<scalar_t>::log(y);
        });
      });
  return out;
}

Tensor& special_xlog1py_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "xlog1py",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t y) -> scalar_t {
          if (at::_isnan(y)) {
            return NAN;
          }
          if (x == 0) {
            return 0;
          }
          return x * Numerics<scalar_t>::log1p(y);
        });
      });
  return out;
}

Tensor& special_zeta_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "zeta", [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t q) -> scalar_t {
          return zeta<scalar_t>(x, q);
        });
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
