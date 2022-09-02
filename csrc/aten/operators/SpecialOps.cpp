#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/Math.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "Random.h"

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

Tensor& special_erfcx_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "erfcx", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return calc_erfcx(a); });
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
