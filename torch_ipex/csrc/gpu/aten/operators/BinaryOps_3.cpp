#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AtenIpexTypeXPU.h>

#include <core/DPCPP.h>
#include <utils/General.h>
#include <utils/Pointwise.h>

#include "Loops.h"


using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
typename std::enable_if<IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t), void>::type
__and___out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (xpu::dpcpp::TensorImpl_Unwrap(result) ==
      xpu::dpcpp::TensorImpl_Unwrap(self)) {
    xpu::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        result, other, TensorBitAndOp<scalar_t>());
  } else {
    at::AtenIpexTypeXPU::resize_as_(result, self, c10::nullopt);
    xpu::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        result, self, other, TensorBitAndOp<scalar_t>());
  }
}

template <typename scalar_t>
typename std::enable_if<!(IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t)), void>::
    type
    __and___out(Tensor& result, const Tensor& self, const Tensor& other) {}

template <typename scalar_t>
typename std::enable_if<IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t), void>::type
__or___out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (xpu::dpcpp::TensorImpl_Unwrap(result) ==
      xpu::dpcpp::TensorImpl_Unwrap(self)) {
    xpu::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        result, other, TensorBitOrOp<scalar_t>());
  } else {
    at::AtenIpexTypeXPU::resize_as_(result, self, c10::nullopt);
    xpu::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        result, self, other, TensorBitOrOp<scalar_t>());
  }
}

template <typename scalar_t>
typename std::enable_if<!(IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t)), void>::
    type
    __or___out(Tensor& result, const Tensor& self, const Tensor& other) {}

template <typename scalar_t>
typename std::enable_if<IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t), void>::type
__xor___out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (xpu::dpcpp::TensorImpl_Unwrap(result) ==
      xpu::dpcpp::TensorImpl_Unwrap(self)) {
    xpu::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        result, other, TensorBitXorOp<scalar_t>());
  } else {
    at::AtenIpexTypeXPU::resize_as_(result, self, c10::nullopt);
    xpu::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        result, self, other, TensorBitXorOp<scalar_t>());
  }
}

template <typename scalar_t>
typename std::enable_if<!(IS_BOOL(scalar_t) || IS_INTEGRAL(scalar_t)), void>::
    type
    __xor___out(Tensor& result, const Tensor& self, const Tensor& other) {}

template <typename...>
class minimum_kernel_bool {};
template <typename...>
class minimum_kernel_interge {};
template <typename...>
class minimum_kernel_float {};

void minimum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter<minimum_kernel_bool<bool>>(iter, [](bool a, bool b) -> bool {
            return a && b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter<minimum_kernel_interge<scalar_t>>(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return std::min(a, b);
    });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "min_elementwise_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter<minimum_kernel_float<scalar_t>>(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            if (a != a) {
              return a;
            } else if (b != b) {
              return b;
            } else {
              return Numerics<scalar_t>::min(a, b);
            }
    });
    });
  }
}

template <typename...>
class maximum_kernel_bool {};
template <typename...>
class maximum_kernel_interge {};
template <typename...>
class maximum_kernel_float {};

void maximum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter<maximum_kernel_bool<bool>>(iter, [](bool a, bool b) -> bool {
            return a || b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "maximum_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter<maximum_kernel_interge<scalar_t>>(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return std::max(a, b);
    });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "max_elementwise_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter<maximum_kernel_float<scalar_t>>(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            if (a != a) {
              return a;
            } else if (b != b) {
              return b;
            } else {
              return Numerics<scalar_t>::max(a, b);
            }
    });
    });
  }
}
} // namespace impl


Tensor& minimum_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "minimum does not support complex inputs.");

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::minimum_kernel(iter);
  return result;
}

Tensor minimum(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "minimum does not support complex inputs.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::minimum_kernel(iter);
  return iter.output();
}

// binary min, alias for minimum
Tensor& min_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::minimum_out(result, self, other);
}

Tensor min(const Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::minimum(self, other);
}

Tensor& maximum_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "maximum does not support complex inputs.");

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::maximum_kernel(iter);
  return result;
}

Tensor maximum(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "maximum does not support complex inputs.");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::maximum_kernel(iter);
  return iter.output();
}

// binary max, alias for maximum
Tensor& max_out(Tensor& result, const Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::maximum_out(result, self, other);
}

Tensor max(const Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::maximum(self, other);
}

Tensor& bitwise_and_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, self.scalar_type(), "__and___out", [&]() {
        impl::__and___out<scalar_t>(result, self, other);
      });
  return result;
}

Tensor& bitwise_or_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, self.scalar_type(), "__or___out", [&]() {
        impl::__or___out<scalar_t>(result, self, other);
      });
  return result;
}

Tensor& bitwise_xor_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, self.scalar_type(), "__xor___out", [&]() {
        impl::__xor___out<scalar_t>(result, self, other);
      });
  return result;
}


Tensor& bitwise_and_out(Tensor& out, const Tensor& self, Scalar other) {
  auto other_ = c10::scalar_to_tensor(other, kXPU);
  // TODO: broadcast
  auto new_other =
      other_.resize_as_(self).fill_(other).toType(self.scalar_type());
  return at::AtenIpexTypeXPU::bitwise_and_out(out, self, new_other);
}

Tensor& bitwise_or_out(Tensor& out, const Tensor& self, Scalar other) {
  auto other_ = c10::scalar_to_tensor(other, kXPU);
  // TODO: broadcast
  auto new_other =
      other_.resize_as_(self).fill_(other).toType(self.scalar_type());
  return at::AtenIpexTypeXPU::bitwise_or_out(out, self, new_other);
}

Tensor& bitwise_xor_out(Tensor& out, const Tensor& self, Scalar other) {
  auto other_ = c10::scalar_to_tensor(other, kXPU);
  // TODO: broadcast
  auto new_other =
      other_.resize_as_(self).fill_(other).toType(self.scalar_type());
  return at::AtenIpexTypeXPU::bitwise_xor_out(out, self, new_other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
