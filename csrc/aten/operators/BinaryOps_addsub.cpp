#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "EltwiseNaiveKer.h"
#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace impl {

void add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "add",
      [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        dpcpp_fast_mode_kernel_with_scalars(
            iter,
            [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; });
      });
}

void sub_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  return add_kernel_dpcpp(iter, -alpha_scalar);
}

// alpha_check
inline void alpha_check(const TensorIterator& iter, Scalar alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || iter.dtype() == ScalarType::Bool,
      "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(
      isFloatingType(iter.dtype()) || isComplexType(iter.dtype()) ||
          alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating "
      "point number.");
}

// Basic checking for all sub functions.
inline void sub_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.scalar_type() != kBool || other.scalar_type() != kBool,
      "Subtraction, the `-` operator, with two bool tensors is not supported. "
      "Use the `^` or `logical_xor()` operator instead.");
  TORCH_CHECK(
      self.scalar_type() != kBool && other.scalar_type() != kBool,
      "Subtraction, the `-` operator, with a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` "
      "operator instead.");
}

} // namespace impl

namespace AtenIpexTypeXPU {

Tensor& add_out(
    const Tensor& _self,
    const Tensor& _other,
    const Scalar& alpha,
    Tensor& result) {
  if ((!alpha.isComplex()) && 1.0 == alpha.to<float>() &&
      xpu::oneDNN::binary_valid(_self, _other) &&
      IPEX_ANY(xpu::oneDNN::is_onednn_layout, _self, _other)) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_add>(result, _self, _other);
    return result;
  } else {
    result = to_plain_if_needed_(result);
    auto self = to_plain_if_needed(_self);
    auto other = to_plain_if_needed(_other);

    auto iter = TensorIterator::binary_op(result, self, other);
    impl::alpha_check(iter, alpha);
    impl::add_kernel_dpcpp(iter, alpha);
    TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());

    auto smf = _self.suggest_memory_format();
    if (is_channels_last(smf)) {
      if (!result.is_contiguous(smf)) {
        result.contiguous(smf);
      }
    }
    return result;
  }
}

Tensor add(const Tensor& _self, const Tensor& _other, const Scalar& alpha) {
  Tensor result;
  if ((!alpha.isComplex()) && 1.0 == alpha.to<float>() &&
      xpu::oneDNN::binary_valid(_self, _other) &&
      IPEX_ANY(xpu::oneDNN::is_onednn_layout, _self, _other)) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_add>(result, _self, _other);
    return result;
  } else {
    auto self = to_plain_if_needed(_self);
    auto other = to_plain_if_needed(_other);

    auto iter = TensorIterator::binary_op(result, self, other);
    impl::alpha_check(iter, alpha);
    impl::add_kernel_dpcpp(iter, alpha);

    auto smf = _self.suggest_memory_format();
    if (is_channels_last(smf)) {
      if (!(iter.output().is_contiguous(smf))) {
        iter.output().contiguous(smf);
      }
    }
    return iter.output();
  }
}

Tensor& add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::AtenIpexTypeXPU::add_out(self, other, alpha, self);
}

Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::AtenIpexTypeXPU::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::AtenIpexTypeXPU::add_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result) {
  impl::sub_check(self, other);
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::sub_kernel_dpcpp(iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, other, self);
  out = iter.output();
  return AtenIpexTypeXPU::sub_out(other, self, alpha, out);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor add(const Tensor& _self, const Tensor& _other, const Scalar& alpha) {
  Tensor result, self, other;
  if (1.0 == alpha.to<float>() && _self.defined() && _other.defined() &&
      _self.sizes() == _other.sizes() && !is_wrapped_number(_self) &&
      !is_wrapped_number(_other) &&
      (!DPCPPTensorContext::is_plain(_self) ||
       !DPCPPTensorContext::is_plain(_other))) {
    xpu::oneDNN::sum(
        result, {_self.contiguous(), _other.contiguous()}, {1.0, 1.0});
    return result;
  } else if (
      _self.is_quantized() && _self.defined() && _other.defined() &&
      _self.sizes() == _other.sizes() && !is_wrapped_number(_self) &&
      !is_wrapped_number(_other)) { // &&
    Tensor _post = at::empty({1}, _self.options().dtype(at::kFloat));
    _post.fill_(1 / alpha.to<float>());
    result = at::_empty_affine_quantized(
        _self.sizes(),
        _self.options(),
        alpha.to<float>(),
        0,
        _self.suggest_memory_format());
    xpu::oneDNN::bin<dnnl::algorithm::binary_add, dnnl::algorithm::binary_mul>(
        result, _self, _other, _post);
    return result;
  } else {
    self = to_plain_if_needed(_self);
    other = to_plain_if_needed(_other);

    auto iter = TensorIterator::binary_op(result, self, other);
    impl::alpha_check(iter, alpha);
    impl::add_kernel_dpcpp(iter, alpha);
    auto smf = _self.suggest_memory_format();
    if (is_channels_last(smf)) {
      if (!(iter.output().is_contiguous(smf))) {
        iter.output().contiguous(smf);
      }
    }

    return iter.output();
  }
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
