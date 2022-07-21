#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>
#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void sqrt_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "sqrt",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::sqrt(a);
        });
      });
}

Tensor& sqrt_out(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_float_op(result, self);
  sqrt_kernel_xpu(iter);
  return result;
}

void rsqrt_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "rsqrt",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::rsqrt(a);
        });
      });
}

Tensor& rsqrt_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  rsqrt_kernel_xpu(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
