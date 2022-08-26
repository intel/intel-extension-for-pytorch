#include <ATen/Context.h>
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

static void logaddexp_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "logaddexp_xpu", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_fast_mode_kernel_with_scalars(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t {
              if (Numerics<accscalar_t>::isinf(static_cast<accscalar_t>(a)) &&
                  a == b) {
                return a;
              } else {
                scalar_t m = Numerics<scalar_t>::max(a, b);
                return m +
                    Numerics<scalar_t>::log(
                           (scalar_t)(1.0) +
                           Numerics<scalar_t>::exp(
                               -Numerics<scalar_t>::abs(a - b)));
              }
            });
      });
}

static void logaddexp2_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "logaddexp2_xpu", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_fast_mode_kernel_with_scalars(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t {
              if (Numerics<accscalar_t>::isinf(static_cast<accscalar_t>(a)) &&
                  a == b) {
                return a;
              } else {
                scalar_t m = Numerics<scalar_t>::max(a, b);
                return m +
                    Numerics<scalar_t>::log2(
                           (scalar_t)(1.0) +
                           Numerics<scalar_t>::pow(
                               (scalar_t)(2.0),
                               -Numerics<scalar_t>::abs(a - b)));
              }
            });
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
