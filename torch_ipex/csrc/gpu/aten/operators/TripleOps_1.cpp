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

} // impl

Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha) {
  impl::dim_check(self, other, accumu);
  Tensor _self, _other, _accumu, result;
  if (check_has_opaque_and_no_padding({self, other, accumu})) {
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

    auto tar_ctx = AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(tar);

    for (int i = 0; i < inputs.size(); ++i) {
      if (!tar.is_same(inputs[i])) {
        Tensor cur = inputs[i];
        auto cur_ctx =
            AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(cur);
        if (cur_ctx.meta() != tar_ctx.meta()) {
          cur = empty_opaque_tensor(
              tar_ctx.meta(), inputs[i].options(), c10::nullopt);
          AtenIpexTypeDPCPP::DPCPPTensorConvertor::convert(cur, inputs[i]);
        }
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
