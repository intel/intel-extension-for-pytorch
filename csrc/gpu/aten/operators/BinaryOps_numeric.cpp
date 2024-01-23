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

template <typename scalar_t>
struct RemainderOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    scalar_t r = a % b;
    if (!std::is_unsigned<scalar_t>::value && (r != 0) &&
        ((r < 0) != (b < 0))) {
      r += b;
    }
    return r;
  }
};

template <typename scalar_t>
struct RemainderOutFunctor2 {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto mod = Numerics<scalar_t>::fmod(a, b);
    if (!std::is_unsigned<scalar_t>::value && (mod != 0) &&
        ((b < 0) != (mod < 0))) {
      mod += b;
    }
    return mod;
  }
};

Tensor& remainder_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_xpu", [&]() {
      RemainderOutFunctor<scalar_t> f;
      dpcpp_kernel_with_scalars(iter, f);
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "remainder_xpu", [&]() {
          RemainderOutFunctor2<scalar_t> f;
          dpcpp_kernel_with_scalars(iter, f);
        });
  }
  return out;
}

Tensor remainder(const Scalar& self, const Tensor& other) {
  return at::remainder(at::native::wrapped_scalar_tensor(self), other);
}

template <typename scalar_t>
struct FmodOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a % b;
  }
};

template <typename scalar_t>
struct FmodOutFunctor2 {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::fmod(a, b);
  }
};

Tensor& fmod_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_xpu", [&]() {
      FmodOutFunctor<scalar_t> f;
      dpcpp_kernel_with_scalars(iter, f);
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "fmod_xpu", [&]() {
          FmodOutFunctor2<scalar_t> f;
          dpcpp_kernel_with_scalars(iter, f);
        });
  }
  return out;
}

template <typename scalar_t>
struct CopysignOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::copysign(a, b);
  }
};

Tensor& copysign_out(const Tensor& self, const Tensor& other, Tensor& out) {
  /* to handle copysign_out(tensor, scalar) redispatch path. */
  /* by default, scalar will be wrapped as a CPU tensor in default */
  /* catchall implememtation. here convert to XPU lazily. */
  Tensor other_maybe_scalar = other;
  if (other.device().type() == at::kCPU && other.numel() == 1) {
    other_maybe_scalar = other.to("xpu");
  }

  auto iter = TensorIterator::binary_float_op(out, self, other_maybe_scalar);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "copysign_out",
      [&]() {
        CopysignOutFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
