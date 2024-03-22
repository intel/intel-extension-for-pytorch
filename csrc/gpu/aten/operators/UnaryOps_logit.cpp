#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {

namespace AtenIpexTypeXPU {

namespace impl {

template <typename scalar_t, typename T_ACC>
struct logit_kernel_xpu_functor {
  scalar_t operator()(scalar_t x) const {
    const T_ACC x_acc = static_cast<T_ACC>(x);
    return Numerics<scalar_t>::log(x_acc / (T_ACC(1) - x_acc));
  }
};

template <typename scalar_t, typename T_ACC>
struct logit_kernel_xpu_functor_2 {
  scalar_t operator()(scalar_t x) const {
    const T_ACC x_acc = static_cast<T_ACC>(x);
    T_ACC z = x_acc < lo ? lo : (x_acc > hi ? hi : x_acc);
    return Numerics<scalar_t>::log(z / (T_ACC(1) - z));
  }

  logit_kernel_xpu_functor_2(const T_ACC lo, const T_ACC hi) : lo(lo), hi(hi) {}

 private:
  const T_ACC lo;
  const T_ACC hi;
};

void logit_kernel_xpu(TensorIterator& iter, c10::optional<double> eps) {
  Scalar eps_scalar = Scalar(eps ? eps.value() : -1.0);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "logit",
      [&]() {
        using T_ACC = acc_type<scalar_t>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          logit_kernel_xpu_functor<scalar_t, T_ACC> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          logit_kernel_xpu_functor_2<scalar_t, T_ACC> f(lo, hi);
          dpcpp_kernel_for_tensor_iter(iter, f);
        }
      });
}

} // namespace impl

Tensor& logit_out(
    const Tensor& self,
    c10::optional<double> eps,
    Tensor& result) {
  auto iter = TensorIterator::unary_float_op(result, self);
  impl::logit_kernel_xpu(iter, eps);
  iter.cast_outputs();
  return result;
}

Tensor logit(const Tensor& self, c10::optional<double> eps) {
  Tensor result;
  auto iter = TensorIterator::unary_float_op(result, self);
  impl::logit_kernel_xpu(iter, eps);
  return iter.output();
}

Tensor& logit_(Tensor& self, c10::optional<double> eps) {
  return logit_out(self, self, eps);
}

} // namespace AtenIpexTypeXPU
} // namespace at