#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/Math.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct igamma_kernel_xpu_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_igamma(a, b);
  }
};

void igamma_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "igamma_xpu",
      [&]() {
        igamma_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

template <typename scalar_t>
struct igammac_kernel_xpu_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_igammac(a, b);
  }
};

void igammac_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "igammac_xpu",
      [&]() {
        igammac_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

} // namespace impl

Tensor& igamma_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  impl::igamma_kernel_xpu(iter);
  return out;
}

Tensor& igammac_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  impl::igammac_kernel_xpu(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
