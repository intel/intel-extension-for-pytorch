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

static void add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
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

static void sub_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  return add_kernel_dpcpp(iter, -alpha_scalar);
}

// alpha_check
static inline void alpha_check(const TensorIterator& iter, Scalar alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || iter.dtype() == ScalarType::Bool,
      "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(
      isFloatingType(iter.dtype()) || alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating "
      "point number.");
}

// Basic checking for all sub functions.
static inline void sub_check(const Tensor& self, const Tensor& other) {
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
  if (_self.is_xpu() && _other.is_xpu() && 1.0 == alpha.to<float>() &&
      _self.defined() && _other.defined() &&
      _self.scalar_type() == _other.scalar_type() &&
      xpu::oneDNN::is_supported_onednn_dtype(_self) &&
      xpu::oneDNN::is_supported_onednn_dtype(_other) && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      ((_self.is_contiguous() && _other.is_contiguous()) ||
       (_self.is_contiguous(MemoryFormat::ChannelsLast) &&
        _other.is_contiguous(MemoryFormat::ChannelsLast))) &&
      !(DPCPPTensorContext::is_plain(_self) &&
        !DPCPPTensorContext::is_plain(_other) &&
        _self.sizes() != _other.sizes()) &&
      !(is_expandable_to(_self.sizes(), _other.sizes()) &&
        !is_expandable_to(_other.sizes(), _self.sizes())) &&
      !is_wrapped_number(_self) && !is_wrapped_number(_other)) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_add>(result, _self, _other);
    return result;
  } else if (
      _self.is_xpu() && _other.is_xpu() && _self.sizes() == _other.sizes() &&
      ((_self.is_contiguous() && _other.is_contiguous()) ||
       (_self.is_contiguous(MemoryFormat::ChannelsLast) &&
        _other.is_contiguous(MemoryFormat::ChannelsLast))) &&
      _self.scalar_type() == _other.scalar_type()) {
    // propogate block format in case: alpha != 1
    if (!DPCPPTensorContext::is_plain(result) ||
        !DPCPPTensorContext::is_plain(_self) ||
        !DPCPPTensorContext::is_plain(_other)) {
      auto r_ctx = DPCPPTensorContext::get_tensor_ctx(result);
      auto s_ctx = DPCPPTensorContext::get_tensor_ctx(_self);
      auto o_ctx = DPCPPTensorContext::get_tensor_ctx(_other);
      auto r_md = r_ctx.meta();
      auto s_md = s_ctx.meta();
      auto o_md = o_ctx.meta();

      auto tar_ctx = !r_ctx.is_plain()
          ? r_ctx
          : (!s_ctx.is_plain() ? s_ctx : (!o_ctx.is_plain() ? o_ctx : s_ctx));
      auto tar_md = tar_ctx.meta();

      if (r_md != tar_md) {
        auto _res = at::AtenIpexTypeXPU::empty_opaque_tensor(
            tar_md, result.options(), c10::nullopt);

        if (result.is_same(_self))
          xpu::oneDNN::reorder(result, _res);

        // result is alias, have to write back
        auto tar_r_ctx = DPCPPTensorContext::release_tensor_ctx(_res);
        DPCPPTensorContext::set_tensor_ctx(result, std::move(tar_r_ctx));
      }

      // avoid redundant reorder in inplace case
      if (!result.is_same(_self) && s_md != tar_md) {
        self = at::AtenIpexTypeXPU::empty_opaque_tensor(
            tar_md, _self.options(), c10::nullopt);
        xpu::oneDNN::reorder(_self, self);
      }

      if (o_md != tar_md) {
        other = at::AtenIpexTypeXPU::empty_opaque_tensor(
            tar_md, _other.options(), c10::nullopt);
        xpu::oneDNN::reorder(_other, other);
      }
    }

    IPEX_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        at::ScalarType::Half,
        result.scalar_type(),
        "eltwise_binary_naive::add",
        [&]() {
          const auto op = AddNaiveOp<scalar_t>(alpha.to<scalar_t>());
          int nelem = !DPCPPTensorContext::is_plain(result)
              ? DPCPPTensorContext::get_tensor_ctx(result).padded_size()
              : prod_intlist(result.sizes());
          eltwise_binary_naive_kernel(
              result.data_ptr<scalar_t>(),
              self.data_ptr<scalar_t>(),
              other.data_ptr<scalar_t>(),
              nelem,
              op);
        });
    return result;
  } else {
    // loops
    // use inplace conversion not to break alias property "Tensor& result"
    result = to_plain_if_needed_(result);
    self = to_plain_if_needed(_self);
    other = to_plain_if_needed(_other);
  }

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::add_kernel_dpcpp(iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());

  return result;
}

Tensor add(const Tensor& _self, const Tensor& _other, Scalar alpha) {
  Tensor result, self, other;
  if (1.0 == alpha.to<float>() && _self.defined() && _other.defined() &&
      xpu::oneDNN::is_supported_onednn_dtype(_self) &&
      xpu::oneDNN::is_supported_onednn_dtype(_other) && _self.dim() > 0 &&
      _other.dim() > 0 && _self.dim() == _other.dim() &&
      _self.is_contiguous() && _other.is_contiguous() &&
      !(DPCPPTensorContext::is_plain(_self) &&
        !DPCPPTensorContext::is_plain(_other) &&
        _self.sizes() != _other.sizes()) &&
      !(is_expandable_to(_self.sizes(), _other.sizes()) &&
        !is_expandable_to(_other.sizes(), _self.sizes())) &&
      !is_wrapped_number(_self) && !is_wrapped_number(_other)) {
    xpu::oneDNN::bin<dnnl::algorithm::binary_add>(result, _self, _other);
    return result;
  } else {
    self = to_plain_if_needed(_self);
    other = to_plain_if_needed(_other);
  }

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::add_kernel_dpcpp(iter, alpha);
  return iter.output();
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
        MemoryFormat::Contiguous);
    xpu::oneDNN::bin<dnnl::algorithm::binary_add, dnnl::algorithm::binary_mul>(
        result, _self, _other, _post);
    return result;
  } else {
    self = to_plain_if_needed(_self);
    other = to_plain_if_needed(_other);
  }

  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::add_kernel_dpcpp(iter, alpha);
  return iter.output();
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
