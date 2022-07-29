#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& remainder_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(out, self, other);
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_xpu", [&]() {
      dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        scalar_t r = a % b;
        if (!dpl::is_unsigned<scalar_t>::value && (r != 0) &&
            ((r < 0) != (b < 0))) {
          r += b;
        }
        return r;
      });
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "remainder_xpu", [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                auto mod = Numerics<scalar_t>::fmod(a, b);
                if (!dpl::is_unsigned<scalar_t>::value && (mod != 0) &&
                    ((b < 0) != (mod < 0))) {
                  mod += b;
                }
                return mod;
              });
        });
  }
  return out;
}

Tensor remainder(const Scalar& self, const Tensor& other) {
  return at::remainder(at::native::wrapped_scalar_tensor(self), other);
}

Tensor& fmod_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(out, self, other);
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_xpu", [&]() {
      dpcpp_kernel_with_scalars(
          iter, [](scalar_t a, scalar_t b) -> scalar_t { return a % b; });
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND(
        kHalf, iter.common_dtype(), "fmod_xpu", [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return Numerics<scalar_t>::fmod(a, b);
              });
        });
  }
  return out;
}

IPEX_BINARY_LOOPS_FUNC_FLOAT_ALL_SCALAR(
    copysign_out,
    Numerics<scalar_t>::copysign,
    binary_float_op)

} // namespace AtenIpexTypeXPU
} // namespace at
