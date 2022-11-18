#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {

namespace AtenIpexTypeXPU {

namespace impl {

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
          dpcpp_kernel_for_tensor_iter(iter, [](scalar_t x) -> scalar_t {
            const T_ACC x_acc = static_cast<T_ACC>(x);
            return Numerics<scalar_t>::log(x_acc / (T_ACC(1) - x_acc));
          });
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          dpcpp_kernel_for_tensor_iter(iter, [lo, hi](scalar_t x) -> scalar_t {
            const T_ACC x_acc = static_cast<T_ACC>(x);
            T_ACC z = x_acc < lo ? lo : (x_acc > hi ? hi : x_acc);
            return Numerics<scalar_t>::log(z / (T_ACC(1) - z));
          });
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