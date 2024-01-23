#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct HeavisideOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a == 0 ? b : static_cast<scalar_t>(a > 0);
  }
};

Tensor& heaviside_out(const Tensor& self, const Tensor& values, Tensor& out) {
  TORCH_CHECK(
      !self.is_complex() && !values.is_complex(),
      "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(
      self.dtype() == values.dtype(),
      "heaviside is not yet implemented for tensors with different dtypes.");

  auto iter = TensorIterator::binary_op(out, self, values);
  IPEX_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, kBFloat16, iter.dtype(), "heaviside", [&]() {
        HeavisideOutFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct hypot_out_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::hypot(a, b);
  }
};

Tensor& hypot_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "hypot",
      [&]() {
        hypot_out_functor<scalar_t> f;
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct NextafterOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::nextafter(a, b);
  }
};

Tensor& nextafter_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, iter.common_dtype(), "nextafter", [&]() {
        NextafterOutFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
  return out;
}

template <typename scalar_t, typename T_ACC>
struct LogitBackwardOutFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    const T_ACC dy_acc = static_cast<T_ACC>(dy);
    const T_ACC x_acc = static_cast<T_ACC>(x);
    return (x_acc < T_ACC(0) || x_acc > T_ACC(1))
        ? std::numeric_limits<T_ACC>::quiet_NaN()
        : dy_acc / (x_acc * (T_ACC(1) - x_acc));
  }
};

template <typename scalar_t, typename T_ACC>
struct LogitBackwardOutFunctor2 {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    const T_ACC dy_acc = static_cast<T_ACC>(dy);
    const T_ACC x_acc = static_cast<T_ACC>(x);
    return (x_acc < lo || x_acc > hi) ? T_ACC(0)
                                      : dy_acc / (x_acc * (T_ACC(1) - x_acc));
  }
  LogitBackwardOutFunctor2(const T_ACC lo_, const T_ACC hi_)
      : lo(lo_), hi(hi_) {}

 private:
  const T_ACC lo;
  const T_ACC hi;
};

at::Tensor& logit_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::optional<double> eps,
    at::Tensor& grad_input) {
  Scalar eps_scalar = Scalar(eps ? eps.value() : -1.0);
  auto iter = TensorIterator::binary_op(grad_input, grad_output, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "logit_xpu",
      [&]() {
        using T_ACC = acc_type<scalar_t>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          LogitBackwardOutFunctor<scalar_t, T_ACC> f;
          dpcpp_kernel_with_scalars(iter, f);
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          LogitBackwardOutFunctor2<scalar_t, T_ACC> f(lo, hi);
          dpcpp_kernel_with_scalars(iter, f);
        }
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
