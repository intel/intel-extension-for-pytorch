#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void mul_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
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
  if (isIntegralType(iter.dtype(), false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div", [&] {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "div",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t { return a / b; });
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
  Tensor _self = self, _other = other;
  const auto ndim = _self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }

  if (_self.defined() && _other.defined() && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      _self.scalar_type() == _other.scalar_type() &&
      xpu::oneDNN::is_supported_onednn_dtype(_self) &&
      xpu::oneDNN::is_supported_onednn_dtype(_other) &&
      ((_self.is_contiguous() && _other.is_contiguous()) ||
       (_self.is_contiguous(cl_tag) && _other.is_contiguous(cl_tag))) &&
      (!DPCPPTensorContext::is_plain(_self) ||
       !DPCPPTensorContext::is_plain(_other)) &&
      _self.sizes() == _other.sizes()) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_div>(result, self, other);
  } else {
    result = to_plain_if_needed_(result);
    _self = to_plain_if_needed(self);
    _other = to_plain_if_needed(other);
    auto iter = TensorIterator::binary_op(result, _self, _other);
    impl::div_kernel_dpcpp(iter);
    auto smf = self.suggest_memory_format();
    if (is_channels_last(smf)) {
      if (!result.is_contiguous(smf)) {
        result.contiguous(smf);
      }
    }
  }
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result, _self = self, _other = other;
  const auto ndim = _self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }

  if (_self.defined() && _other.defined() && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      _self.scalar_type() == _other.scalar_type() &&
      xpu::oneDNN::is_supported_onednn_dtype(_self) &&
      xpu::oneDNN::is_supported_onednn_dtype(_other) &&
      ((_self.is_contiguous() && _other.is_contiguous()) ||
       (_self.is_contiguous(cl_tag) && _other.is_contiguous(cl_tag))) &&
      (!DPCPPTensorContext::is_plain(_self) ||
       !DPCPPTensorContext::is_plain(_other)) &&
      _self.sizes() == _other.sizes()) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_div>(result, self, other);
    return result;
  } else {
    result = to_plain_if_needed_(result);
    _self = to_plain_if_needed(self);
    _other = to_plain_if_needed(other);
    auto iter = TensorIterator::binary_op(result, _self, _other);
    impl::div_kernel_dpcpp(iter);
    auto smf = self.suggest_memory_format();
    if (is_channels_last(smf)) {
      if (!iter.output().is_contiguous(smf)) {
        iter.output().contiguous(smf);
      }
    }
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
  auto iter = TensorIterator::binary_op(result, self, other);
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
