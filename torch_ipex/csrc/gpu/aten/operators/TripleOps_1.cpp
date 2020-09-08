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
      "mul_add_",
      [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<SyclOpMulAdd>(
            iter, [=](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
              return a * b + alpha * c;
            });
      });
}
}

Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha) {
  Tensor _self, _other, _accumu;
  Tensor result = at::empty({0}, accumu.options());
  // TODO: support to propagate block fmt
#if 0
  if ((!DPCPPTensorContext::is_plain(self) ||
      !DPCPPTensorContext::is_plain(other) ||
      !DPCPPTensorContext::is_plain(accumu)) &&
      (self_ctx.padded_size() == self.numel() &&
       other_ctx.padded_size() == other.numel() &&
       bias_ctx.padded_size() == accumu.numel())
     ) {
    // reorder blocked format for plain format
    // self
    // other
    // bias

  } else {
#endif
    _self = to_plain_if_needed(self);
    _other = to_plain_if_needed(other);
    _accumu = to_plain_if_needed(accumu);
#if 0
  }
#endif

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
