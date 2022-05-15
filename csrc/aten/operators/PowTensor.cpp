#include <core/TensorImplUtils.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void pow_tensor_scalar_kernel(TensorIterator& iter, const Scalar& exp_scalar);

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "pow_xpu", [&]() {
        if (iter.is_cpu_scalar(1)) {
          const auto base = iter.scalar_value<scalar_t>(1);
          iter.remove_operand(1);
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t exp) -> scalar_t {
            return Numerics<scalar_t>::pow(base, exp);
          });
        } else if (iter.is_cpu_scalar(2)) {
          const auto exp = iter.scalar_value<scalar_t>(2);
          iter.remove_operand(2);
          at::AtenIpexTypeXPU::pow_tensor_scalar_kernel(iter, exp);
        } else {
          dpcpp_kernel_for_tensor_iter(
              iter, [](scalar_t base, scalar_t exp) -> scalar_t {
                return Numerics<scalar_t>::pow(base, exp);
              });
        }
      });
}

Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto iter = TensorIterator::binary_op(result, base, exp);
  pow_tensor_tensor_kernel(iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Scalar& base, const Tensor& exp) {
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else {
    at::AtenIpexTypeXPU::pow_out(result, wrapped_scalar_tensor(base), exp);
  }
  return result;
}

Tensor pow(const Tensor& base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty({0}, base.options().dtype(dtype));
  return at::AtenIpexTypeXPU::pow_out(result, base, exp);
}

} // namespace AtenIpexTypeXPU
} // namespace at
