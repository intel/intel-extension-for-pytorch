#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>

#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"
#include "Resize.h"
#include "utils/CustomOperatorRegistration.h"
#include "utils/logging.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename func_t>
static inline Tensor& unary_op_impl_with_complex_to_float_out(
    Tensor& result,
    const Tensor& self,
    const func_t& fn,
    bool promotes_integer_to_float) {
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(
        canCast(float_type, result.scalar_type()),
        "result type ",
        float_type,
        " can't be cast to the desired output type ",
        result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    Tensor complex_result = at::empty({0}, self.options());
    auto self_ = at::AtenIpexTypeXPU::to_plain_if_needed(self);
    auto iter = TensorIterator::unary_op(complex_result, self_);
    fn(iter);

    // Copies the complex result to the actual result and returns it
    resize_output(result, complex_result.sizes());
    result.copy_(at::real(complex_result));
    return result;
  }

  if (promotes_integer_to_float) {
    result = at::AtenIpexTypeXPU::to_plain_if_needed_(result);
    auto self_ = at::AtenIpexTypeXPU::to_plain_if_needed(self);
    auto iter = TensorIterator::unary_float_op(result, self_);
    fn(iter);
    iter.cast_outputs();
    return result;
  }

  // abs kernel
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_abs>(
      TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
        fn(iter);
      });
}

template <typename T>
static T abs_impl(T v) {
  return Numerics<T>::abs(v);
}

template <typename scalar_t>
struct abs_kernel_functor {
  scalar_t operator()(scalar_t a) const {
    return abs_impl<scalar_t>(a);
  }
};

void abs_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    IPEX_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      abs_kernel_functor<opmath_t> f;
      dpcpp_kernel_for_tensor_iter(iter, f);
    });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs",
        [&]() {
          abs_kernel_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
}

template <typename scalar_t>
struct angle_kernel_functor {
  scalar_t operator()(scalar_t a) const {
    return at::AtenIpexTypeXPU::angle_impl(a);
  }
};

void angle_kernel(TensorIteratorBase& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "angle", [&]() {
        angle_kernel_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

} // namespace impl

Tensor& abs_out(const Tensor& self, Tensor& result) {
  return impl::unary_op_impl_with_complex_to_float_out(
      result, self, impl::abs_kernel, /*promotes_integer_to_float=*/false);
}

Tensor& angle_out(const Tensor& self, Tensor& result) {
  return impl::unary_op_impl_with_complex_to_float_out(
      result, self, impl::angle_kernel, /*promotes_integer_to_float=*/true);
}

Tensor angle(const Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return at::angle_out(result, self);
  }
  Tensor result;
  auto iter = TensorIterator::unary_float_op(result, self);
  impl::angle_kernel(iter);
  return iter.output();
}

Tensor nan_to_num(
    const Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  auto result = at::empty_like(self);
  return at::nan_to_num_out(result, self, nan, pos_inf, neg_inf);
}

Tensor& nan_to_num_(
    Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  return at::nan_to_num_out(self, self, nan, pos_inf, neg_inf);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("angle", TORCH_FN((&at::AtenIpexTypeXPU::angle)));
  m.impl("angle.out", TORCH_FN((&at::AtenIpexTypeXPU::angle_out)));
}

} // namespace
