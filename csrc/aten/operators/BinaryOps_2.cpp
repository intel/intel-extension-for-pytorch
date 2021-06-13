#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <runtime/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"
#include <oneDNN/oneDNN.h>

#include "Loops.h"


using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// Note: dpcpp compiler does not support uname type in template.
class SyclOpMul {};
class SyclOpDiv {};

static void mul_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "mul",
      [&]() {
        dpcpp_kernel_with_scalars<SyclOpMul>(
          iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; });
      });
}

static void div_kernel_dpcpp(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div", [&] {
      dpcpp_kernel_with_scalars<SyclOpDiv>(
        iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "div",
        [&]() {
          dpcpp_kernel_with_scalars<SyclOpDiv>(
            iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
        });
  }
}

} // namespace impl

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(
      result,
      self,
      other);
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
  return at::AtenIpexTypeXPU::mul(self, at::wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeXPU::mul_(self, at::wrapped_scalar_tensor(other));
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor _self = self, _other = other;
  if (_self.defined() && _other.defined() && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      xpu::oneDNN::is_supported_dtype_in_binary(
          _self.scalar_type(), _other.scalar_type()) &&
      _self.is_contiguous() && _other.is_contiguous() &&
      !(DPCPPTensorContext::is_plain(_self) &&
        !DPCPPTensorContext::is_plain(_other) &&
        _self.sizes() != _other.sizes()) &&
      !(is_expandable_to(_self.sizes(), _other.sizes()) &&
      !is_expandable_to(_other.sizes(), _self.sizes())) &&
      !at::is_wrapped_number(_self) && !at::is_wrapped_number(_other)) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_div>(result, self, other);
  } else {
    auto iter = TensorIterator::binary_op(result, self, other);
    impl::div_kernel_dpcpp(iter);
  }
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result, _self = self, _other = other;
  if (_self.defined() && _other.defined() && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      xpu::oneDNN::is_supported_dtype_in_binary(
          _self.scalar_type(), _other.scalar_type()) &&
      _self.is_contiguous() && _other.is_contiguous() &&
      !(DPCPPTensorContext::is_plain(_self) &&
        !DPCPPTensorContext::is_plain(_other) &&
        _self.sizes() != _other.sizes()) &&
      !(is_expandable_to(_self.sizes(), _other.sizes()) &&
      !is_expandable_to(_other.sizes(), _self.sizes())) &&
      !at::is_wrapped_number(_self) && !at::is_wrapped_number(_other)) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_div>(result, self, other);
    return result;
  } else {
    auto iter = TensorIterator::binary_op(result, self, other);
    impl::div_kernel_dpcpp(iter);
    return iter.output();
  }
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::div_out(self, self, other);
}

Tensor& floor_divide_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto iter = TensorIterator::binary_op(
      result,
      self,
      other);
  impl::div_kernel_dpcpp(iter);
  if (result.is_floating_point()) {
    result.trunc_();
  }
  return result;
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::div_kernel_dpcpp(iter);

  auto out = iter.output();
  if (out.is_floating_point()) {
    out.trunc_();
  }
  return out;
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::floor_divide_out(self, self, other);
}
} // namespace AtenIpexTypeXPU
} // namespace at
