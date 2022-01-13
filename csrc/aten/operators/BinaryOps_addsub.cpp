#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <intrinsic/ipex_intrinsic.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

#include "EltwiseNaiveKer.h"
#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace impl {

void add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "add",
      [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        dpcpp_kernel_with_scalars(
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
      isFloatingType(iter.dtype()) || alpha.isIntegral(true),
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

template <class T>
class AddNaiveOp {
 public:
  AddNaiveOp(T alpha) : alpha_(alpha) {}

  void operator()(T* res, T* op1, T* op2) const {
    *res = *op1 + alpha_ * (*op2);
  }

 private:
  T alpha_;
};

Tensor& add_out(
    Tensor& result,
    const Tensor& _self,
    const Tensor& _other,
    Scalar alpha) {
  Tensor self = _self, other = _other;
  const auto ndim = _self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }
  if (_self.is_xpu() && _other.is_xpu() && 1.0 == alpha.to<float>() &&
      _self.defined() && _other.defined() &&
      _self.scalar_type() == _other.scalar_type() &&
      xpu::oneDNN::is_supported_onednn_dtype(_self) &&
      xpu::oneDNN::is_supported_onednn_dtype(_other) && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      /* Herein, still use actual memory format to do judgement,
       * because suggest memory format may be not the same as
       * actual memory format. If use suggest memory format for
       * onednn pass judgement, may cause reorder for non-contiguous
       * tensor. However, for non-contigous tensor should use
       * TensorIterator pass below. */
      ((_self.is_contiguous() && _other.is_contiguous()) ||
       (_self.is_contiguous(cl_tag) && _other.is_contiguous(cl_tag))) &&
      !is_wrapped_number(_self) && !is_wrapped_number(_other) &&
      (((!DPCPPTensorContext::is_plain(_self) ||
         !DPCPPTensorContext::is_plain(_other)) &&
        _self.sizes() == _other.sizes()) ||
       (_self.sizes() != _other.sizes() &&
        is_expandable_to(_other.sizes(), _self.sizes())))) {
    /* If the following conditions are satisfied, then oneDNN path will be
     selected:
     * 1. _self and _other should be xpu tensor and be defined.
     * 2. res = _self + alpha * _other; the scalar alpha should be equal to 1.0.
     * 3. _self and _other should be in the same datatype.
     * 4. the datatype should be supported by oneDNN primitive.
     * 5. dim of _self and _other should be equal and must be larger than 0.
     * 6. _self and _other should be contiguous or channel-last contiguous.
     * 7. _self or _other should not be scalar (wrapped tensor).
     * 8. _self or _other is block format and should not involve broadcast,
          or involved in broadcast when _other is expandable to _self (temporary
     decision).
     * TODO: Currently, DPCPP binary ops for tensor broadcast ([4,16,16,512] +
     * [4,1,1,512]) are in poor efficiency. So for these cases, we still use
     * oneDNN path. In the future, we will optimize the tensor broadcast cases
     * and use DPCPP binary ops all the time except the blocked format cases. */
    xpu::oneDNN::bin<dnnl::algorithm::binary_add>(result, _self, _other);
    return result;
  } else {
    // loops
    // use inplace conversion not to break alias property "Tensor& result"
    result = to_plain_if_needed_(result);
    self = to_plain_if_needed(_self);
    other = to_plain_if_needed(_other);

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

Tensor add(const Tensor& _self, const Tensor& _other, Scalar alpha) {
  Tensor result, self, other;
  const auto ndim = _self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }
  if (1.0 == alpha.to<float>() && _self.defined() && _other.defined() &&
      xpu::oneDNN::is_supported_onednn_dtype(_self) &&
      xpu::oneDNN::is_supported_onednn_dtype(_other) && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      ((_self.is_contiguous() && _other.is_contiguous()) ||
       (_self.is_contiguous(cl_tag) && _other.is_contiguous(cl_tag))) &&
      !is_wrapped_number(_self) && !is_wrapped_number(_other) &&
      (((!DPCPPTensorContext::is_plain(_self) ||
         !DPCPPTensorContext::is_plain(_other)) &&
        _self.sizes() == _other.sizes()) ||
       (_self.sizes() != _other.sizes() &&
        is_expandable_to(_other.sizes(), _self.sizes())))) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_add>(result, _self, _other);
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

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::AtenIpexTypeXPU::add_out(self, self, other, alpha);
}

Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeXPU::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeXPU::add_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  impl::sub_check(self, other);
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::sub_kernel_dpcpp(iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  impl::sub_check(self, other);
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::sub_kernel_dpcpp(iter, alpha);
  return iter.output();
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::AtenIpexTypeXPU::sub_out(self, self, other, alpha);
}

Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
  return at::AtenIpexTypeXPU::sub(other, self, alpha);
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeXPU::sub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeXPU::sub_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeXPU::rsub(self, wrapped_scalar_tensor(other), alpha);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor add(const Tensor& _self, const Tensor& _other, Scalar alpha) {
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
