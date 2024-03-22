#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

struct minimum_kernel_functor {
  bool operator()(bool a, bool b) const {
    return a && b;
  }
};

template <typename scalar_t>
struct minimum_kernel_functor_2 {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::min(a, b);
  }
};

void minimum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    minimum_kernel_functor f;
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, f);
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "min_elementwise_dpcpp",
        [&]() {
          minimum_kernel_functor_2<scalar_t> f;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
        });
  }
}

} // namespace impl

Tensor& minimum_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "minimum does not support complex inputs.");

  return binary_out_template<dnnl::algorithm::binary_min>(
      TensorIterator::binary_op,
      result,
      self,
      other,
      [=](TensorIteratorBase& iter) { impl::minimum_kernel(iter); });
}

Tensor minimum(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "minimum does not support complex inputs.");
  Tensor result;
  return binary_out_template<dnnl::algorithm::binary_min>(
      TensorIterator::binary_op,
      result,
      self,
      other,
      [=](TensorIteratorBase& iter) { impl::minimum_kernel(iter); });
}

template <typename scalar_t>
struct fmin_out_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::fmin(a, b);
  }
};

Tensor& fmin_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  if (isFloatingType(iter.common_dtype())) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmin",
        [&]() {
          fmin_out_functor<scalar_t> f;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
        });
  } else {
    impl::minimum_kernel(iter);
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
