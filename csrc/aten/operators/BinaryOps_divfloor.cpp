#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "comm/Numerics.h"
#include "comm/zmath.h"

#include "oneapi/dpl/cmath"
namespace dpl = oneapi::dpl;

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void div_trunc_kernel(TensorIterator& iter);

namespace impl {

static void div_floor_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == kByte) {
    return at::AtenIpexTypeXPU::div_trunc_kernel(iter);
  } else if (isIntegralType(iter.dtype(), false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div_floor_dpcpp", [&] {
      dpcpp_kernel_with_scalars(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        if ((a < 0) != (b < 0)) {
          const auto quot = a / b;
          const auto rem = a % b;
          return rem ? quot - 1 : quot;
        }

        return a / b;
      });
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "div_floor_dpcpp",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [](scalar_t a, scalar_t b) -> scalar_t {
                if (DPCPP_UNLIKELY(b == 0)) {
                  // Divide by zero: return standard IEEE result
                  return a / b;
                }

                auto mod = Numerics<scalar_t>::fmod(a, b);
                auto div = (a - mod) / b;
                if ((mod != 0) && (b < 0) != (mod < 0)) {
                  div -= scalar_t(1);
                }

                scalar_t floordiv;
                if (div != 0) {
                  floordiv = Numerics<scalar_t>::floor(div);
                  if (div - floordiv > scalar_t(0.5)) {
                    floordiv += scalar_t(1.0);
                  }
                } else {
                  floordiv = Numerics<double>::copysign(
                      0.0, scalar_cast<double>(a / b));
                }
                return floordiv;
              });
        });
  }
}

} // namespace impl

void div_trunc_kernel(TensorIterator& iter);

Tensor& floor_divide_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  TORCH_WARN_ONCE(
      "floor_divide is deprecated, and will be removed in a future version of pytorch. "
      "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
      "This results in incorrect rounding for negative values.\n"
      "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
      "or for actual floor division, use torch.div(a, b, rounding_mode='floor').");
  // FIXME: Not actually doing floor division (#43874)
  auto iter = TensorIterator::binary_op(result, self, other);
  at::AtenIpexTypeXPU::div_trunc_kernel(iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  TORCH_WARN_ONCE(
      "floor_divide is deprecated, and will be removed in a future version of pytorch. "
      "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
      "This results in incorrect rounding for negative values.\n"
      "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
      "or for actual floor division, use torch.div(a, b, rounding_mode='floor').");
  // FIXME: Not actually doing floor division (#43874)
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  at::AtenIpexTypeXPU::div_trunc_kernel(iter);
  return iter.output();
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::floor_divide_out(self, self, other);
}

void div_floor_kernel(TensorIterator& iter) {
  impl::div_floor_kernel_dpcpp(iter);
}

} // namespace AtenIpexTypeXPU
} // namespace at
