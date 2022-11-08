#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/Math.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void digamma_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "digamma_xpu",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t a) -> scalar_t { return calc_digamma(a); });
      });
}

void igamma_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "igamma_xpu", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t {
              return calc_igamma(a, b);
            });
      });
}

void igammac_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "igammac_xpu", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t {
              return calc_igammac(a, b);
            });
      });
}

void trigamma_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "trigamma_xpu",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t a) -> scalar_t { return calc_trigamma(a); });
      });
}

void polygamma_kernel_xpu(TensorIterator& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_xpu(iter);
  } else if (n == 1) {
    trigamma_kernel_xpu(iter);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "polygamma_xpu",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t a) -> scalar_t {
            return calc_polygamma(a, n);
          });
        });
  }
}

} // namespace impl

static inline void lgamma_check(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(
      at::isFloatingType(self.scalar_type()),
      "Only support floating data type for now.");
}

Tensor lgamma(const Tensor& self) {
  Tensor out = at::empty_like(self);
  return at::AtenIpexTypeXPU::lgamma_out(self, out);
}

Tensor& lgamma_(Tensor& self) {
  return at::AtenIpexTypeXPU::lgamma_out(self, self);
}

Tensor& lgamma_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "lgamma", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return Numerics<scalar_t>::lgamma(a);
        });
      });
  return out;
}

Tensor& mvlgamma_out(const Tensor& self, int64_t p, Tensor& out) {
  auto output = self.mvlgamma(p);
  TORCH_CHECK(
      at::can_cast(output.scalar_type(), out.scalar_type()),
      "mvlgamma: result type ",
      self.scalar_type(),
      " can't be cast to the desired output type ",
      output.scalar_type());
  at::native::resize_output(out, output.sizes());
  return out.copy_(output);
}

Tensor digamma(const Tensor& self) {
  Tensor result = !at::isFloatingType(self.scalar_type())
      ? at::empty(self.sizes(), self.options().dtype(at::kFloat))
      : at::empty_like(self);
  auto iter = TensorIterator::unary_float_op(result, self);
  impl::digamma_kernel_xpu(iter);
  return result;
}

Tensor& digamma_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  impl::digamma_kernel_xpu(iter);
  return self;
}

Tensor& digamma_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_float_op(out, self);
  impl::digamma_kernel_xpu(iter);
  return out;
}

Tensor& igamma_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  impl::igamma_kernel_xpu(iter);
  return out;
}

Tensor& igammac_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  impl::igammac_kernel_xpu(iter);
  return out;
}

Tensor polygamma(int64_t n, const Tensor& self) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  Tensor result = !at::isFloatingType(self.scalar_type())
      ? at::empty(self.sizes(), self.options().dtype(at::kFloat))
      : at::empty_like(self);
  auto iter = TensorIterator::unary_float_op(result, self);
  impl::polygamma_kernel_xpu(iter, n);
  return result;
}

Tensor& polygamma_(Tensor& self, int64_t n) {
  auto iter = TensorIterator::unary_float_op(self, self);
  impl::polygamma_kernel_xpu(iter, n);
  return self;
}

Tensor& polygamma_out(Tensor& out, int64_t n, const Tensor& self) {
  auto iter = TensorIterator::unary_float_op(out, self);
  impl::polygamma_kernel_xpu(iter, n);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
