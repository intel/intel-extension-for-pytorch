#include <ATen/Context.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "comm/Numerics.h"
#include "comm/zmath.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename accscalar_t>
struct LogAddExpKernelDpcppFunctor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    if (Numerics<accscalar_t>::isinf(static_cast<accscalar_t>(a_)) &&
        a_ == b_) {
      return a_;
    } else {
      const auto a = static_cast<accscalar_t>(a_);
      const auto b = static_cast<accscalar_t>(b_);
      const auto m = Numerics<accscalar_t>::max(a, b);
      return (
          m +
          Numerics<accscalar_t>::log1p(
              Numerics<accscalar_t>::exp(-Numerics<accscalar_t>::abs(a - b))));
    }
  }
};

static void logaddexp_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "logaddexp_xpu", [&]() {
        using accscalar_t = at::opmath_type<scalar_t>;
        LogAddExpKernelDpcppFunctor<scalar_t, accscalar_t> kfn;
        dpcpp_fast_mode_kernel_with_scalars(iter, kfn);
      });
}

template <typename scalar_t, typename opmath_t>
struct LogAddExp2KernelDpcppFunctor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    if (Numerics<opmath_t>::isinf(static_cast<opmath_t>(a_)) && a_ == b_) {
      return a_;
    } else {
      const auto inv_log_2 = static_cast<opmath_t>(1.0 / c10::ln_2<double>);
      const auto a = static_cast<opmath_t>(a_);
      const auto b = static_cast<opmath_t>(b_);
      const auto m = Numerics<opmath_t>::max(a, b);
      return (
          m +
          Numerics<opmath_t>::log1p(
              Numerics<opmath_t>::exp2(-Numerics<opmath_t>::abs(a - b))) *
              inv_log_2);
    }
  }
};

static void logaddexp2_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "logaddexp2_xpu", [&]() {
        using accscalar_t = at::opmath_type<scalar_t>;
        LogAddExp2KernelDpcppFunctor<scalar_t, accscalar_t> kfn;
        dpcpp_fast_mode_kernel_with_scalars(iter, kfn);
      });
}
} // namespace impl

at::Tensor logaddexp(const at::Tensor& self, const at::Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::logaddexp_kernel_dpcpp(iter);
  return iter.output();
}

at::Tensor& logaddexp_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::logaddexp_kernel_dpcpp(iter);
  return out;
}

at::Tensor logaddexp2(const at::Tensor& self, const at::Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::logaddexp2_kernel_dpcpp(iter);
  return iter.output();
}

at::Tensor& logaddexp2_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::logaddexp2_kernel_dpcpp(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
