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
#ifdef USE_ONEMKL
  lgamma_check(self);
  int64_t n = self.numel();
  Tensor out = at::empty_like(self);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lgamma", [&] {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::vm::lgamma,
        dpcpp_queue,
        n,
        (scalar_t*)self.data_ptr(),
        (scalar_t*)out.data_ptr());
  });
  return out;
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

Tensor& lgamma_(Tensor& self) {
#ifdef USE_ONEMKL
  lgamma_check(self);
  int64_t n = self.numel();
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lgamma_", [&] {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::vm::lgamma,
        dpcpp_queue,
        n,
        (scalar_t*)self.data_ptr(),
        (scalar_t*)self.data_ptr());
  });

  return self;
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
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

static inline void mvlgamma_check(const Tensor& self, int64_t p) {
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "mvlgamma is not implemented for ",
      self.scalar_type());
  TORCH_CHECK(
      (self > 0.5f * (p - 1)).all().item<bool>(),
      "All elements must be greater than (p-1)/2");
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
}

Tensor mvlgamma(const Tensor& self, int64_t p) {
#ifdef USE_ONEMKL
  mvlgamma_check(self, p);
  Tensor range = at::empty({0}, self.options());
  Tensor args = at::arange_out(range, -p / 2. + 0.5, 0.5, 0.5);
  args = args.add(self.unsqueeze(-1));

  return args.lgamma_().sum(-1).add_(
      p * (p - 1) * std::log(Numerics<double>::pi()) / 4.);

#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
#ifdef USE_ONEMKL
  mvlgamma_check(self, p);
  Tensor range = at::empty({0}, self.options());
  Tensor args = at::arange_out(range, -p / 2. + 0.5, 0.5, 0.5);
  args = args.add(self.unsqueeze(-1));

  return self.copy_(args.lgamma_().sum(-1).add_(
      p * (p - 1) * std::log(Numerics<double>::pi()) / 4.));
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

Tensor& mvlgamma_out(const Tensor& self, int64_t p, Tensor& out) {
#ifdef USE_ONEMKL
  auto output = self.mvlgamma(p);
  TORCH_CHECK(
      at::can_cast(output.scalar_type(), out.scalar_type()),
      "mvlgamma: result type ",
      self.scalar_type(),
      " can't be cast to the desired output type ",
      output.scalar_type());
  at::native::resize_output(out, output.sizes());
  return out.copy_(output);
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
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

Tensor& digamma_out(const Tensor& self, Tensor& out) {
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

Tensor& polygamma_out(int64_t n, const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  impl::polygamma_kernel_xpu(iter, n);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
