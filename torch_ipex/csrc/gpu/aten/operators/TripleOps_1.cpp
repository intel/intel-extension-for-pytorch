#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

// Note: dpcpp compiler does not support uname type in template.
class SyclOpMulAdd {};

static void mul_add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mul_add",
      [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<SyclOpMulAdd>(
            iter, [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
              return a * b + alpha * c;
            });
      });
}

// Basic checking for all input tensors.
static inline void dim_check(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu) {
  int64_t self_ndims = self.ndimension();
  int64_t other_ndims = other.ndimension();
  int64_t accumu_ndims = accumu.ndimension();

  TORCH_CHECK(
      self_ndims == other_ndims || other_ndims == accumu_ndims,
      "The dimensions of three inputs tensor not equal is not supported. ");
}

static bool opaque_check(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu) {
  auto self_ctx =
      at::AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(self);
  auto other_ctx =
      at::AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(other);
  auto accumu_ctx =
      at::AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(accumu);
  int64_t self_padded_numel =
      DPCPPTensorContext(nullptr, self_ctx.meta()).padded_size();
  int64_t other_padded_numel =
      DPCPPTensorContext(nullptr, other_ctx.meta()).padded_size();
  int64_t accumu_padded_numel =
      DPCPPTensorContext(nullptr, accumu_ctx.meta()).padded_size();

  if (self_ctx.is_plain() && other_ctx.is_plain() && accumu_ctx.is_plain())
    return false;

  if ((self_padded_numel != 0 && self_padded_numel != self.numel()) ||
      (other_padded_numel != 0 && other_padded_numel != other.numel()) ||
      (accumu_padded_numel != 0 && accumu_padded_numel != accumu.numel()))
    return false;

  return true;
}

} // impl

Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha) {
  impl::dim_check(self, other, accumu);
  Tensor _self, _other, _accumu, result;
  if (impl::opaque_check(self, other, accumu)) {
    std::vector<Tensor> inputs;
    inputs.push_back(self);

    // align shape
    if (self.numel() != other.numel())
      inputs.push_back(other.expand_as(self).contiguous());
    else
      inputs.push_back(other);

    if (self.numel() != accumu.numel())
      inputs.push_back(accumu.expand_as(self).contiguous());
    else
      inputs.push_back(accumu);

    // align format
    std::vector<Tensor> _inputs;

    Tensor tar;
    for (int i = 0; i < inputs.size(); ++i) {
      if (DPCPPTensorConvertor::is_opaque_tensor(inputs[i])) {
        tar = inputs[i];
        break;
      }
    }

    auto tar_ctx = *(static_cast<DPCPPTensorContext*>(
        tar.unsafeGetTensorImpl()->storage().data_ptr().get_context()));

    for (int i = 0; i < inputs.size(); ++i) {
      if (!tar.is_same(inputs[i])) {
        auto cur = empty_opaque_tensor(
            tar_ctx.meta(), inputs[i].options(), c10::nullopt);
        AtenIpexTypeDPCPP::DPCPPTensorConvertor::convert(cur, inputs[i]);
        _inputs.push_back(cur);
      } else {
        _inputs.push_back(tar);
      }
    }
    _self = _inputs.at(0);
    _other = _inputs.at(1);
    _accumu = _inputs.at(2);
    result = empty_opaque_tensor(tar_ctx.meta(), tar.options(), c10::nullopt);
  } else {
    _self = to_plain_if_needed(self);
    _other = to_plain_if_needed(other);
    _accumu = to_plain_if_needed(accumu);
    result = at::empty_like(self);
  }

  auto iter = TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(_self);
  iter.add_input(_other);
  iter.add_input(_accumu);
  iter.build();
  impl::mul_add_kernel_dpcpp(iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
