#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <runtime/Utils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include "comm/Numerics.h"
#include "comm/ATDispatch.h"
#include "comm/Math.h"

#include "Random.h"
#include "Loops.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <utils/oneMKLUtils.h>
#endif


using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

DPCPP_DEF_K1(DigammaOp);
void digamma_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "digamma_xpu", [&]() {
      dpcpp_kernel_for_tensor_iter<DPCPP_K(DigammaOp)>(iter, [=](scalar_t a) -> scalar_t {
          return calc_digamma(a);
      });
  });
}

DPCPP_DEF_K1(TrigammaOp);
void trigamma_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "trigamma_xpu", [&]() {
      dpcpp_kernel_for_tensor_iter<DPCPP_K(TrigammaOp)>(iter, [=](scalar_t a) -> scalar_t {
          return calc_trigamma(a);
      });
  });
}

DPCPP_DEF_K1(PolygammaOp);
void polygamma_kernel_xpu(TensorIterator& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_xpu(iter);
  } else if (n == 1) {
    trigamma_kernel_xpu(iter);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "polygamma_xpu", [&]() {
        dpcpp_kernel_for_tensor_iter<DPCPP_K(PolygammaOp)>(iter, [=](scalar_t a) -> scalar_t {
            return calc_polygamma(int(n), a);
        });
    });
  }
}

} // namespace impl


Tensor lgamma(const Tensor & self) {
#ifdef USE_ONEMKL
  int64_t n = self.numel();
  Tensor out = at::empty_like(self);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lgamma", [&] {
    DPCPP_ONEMKL_SUBMIT(dpcpp_queue, oneapi::mkl::vm::lgamma, dpcpp_queue, n, (scalar_t *)self.data_ptr(), (scalar_t *)out.data_ptr());
  });

  return out;
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

Tensor& lgamma_(Tensor & self) {
#ifdef USE_ONEMKL
  int64_t n = self.numel();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lgamma_", [&] {
    DPCPP_ONEMKL_SUBMIT(dpcpp_queue, oneapi::mkl::vm::lgamma, dpcpp_queue, n, (scalar_t *)self.data_ptr(), (scalar_t *)self.data_ptr());
  });

  return self;
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

static inline void mvlgamma_check(const Tensor& self, int64_t p) {
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "mvlgamma is not implemented for ", self.scalar_type());
  TORCH_CHECK((self > 0.5f * (p - 1)).all().item<bool>(),
              "All elements must be greater than (p-1)/2");
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
}

Tensor mvlgamma(const Tensor & self, int64_t p) {
#ifdef USE_ONEMKL
  mvlgamma_check(self, p);
  Tensor args = native::arange(-p / 2. + 0.5, 0.5, 0.5, self.options());
  args = args.add(self.unsqueeze(-1));

  return args.lgamma_().sum(-1).add_(p * (p - 1) * std::log(M_PI) / 4.);
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
#ifdef USE_ONEMKL
  mvlgamma_check(self, p);
  Tensor args = native::arange(-p / 2. + 0.5, 0.5, 0.5, self.options());
  args = args.add(self.unsqueeze(-1));

  return self.copy_(args.lgamma_().sum(-1).add_(p * (p - 1) * std::log(M_PI) / 4.));
#else
  AT_ERROR("lgamma: oneMKL library not found in compilation");
#endif
}

Tensor digamma(const Tensor& self) {
  Tensor result = at::empty_like(self);
  auto iter = TensorIterator::unary_op(result, self);
  impl::digamma_kernel_xpu(iter);
  return result;
}

Tensor digamma_(Tensor& self) {
  auto iter = TensorIterator::unary_op(self, self);
  impl::digamma_kernel_xpu(iter);
  return self;
}

Tensor digamma_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  impl::digamma_kernel_xpu(iter);
  return out;
}

Tensor polygamma(int64_t n, const Tensor& self) {
  Tensor result = at::empty_like(self);
  auto iter = TensorIterator::unary_op(result, self);
  impl::polygamma_kernel_xpu(iter, n);
  return result;
}

Tensor& polygamma_(Tensor& self, int64_t n) {
  auto iter = TensorIterator::unary_op(self, self);
  impl::polygamma_kernel_xpu(iter, n);
  return self;
}

Tensor& polygamma_out(Tensor& out, int64_t n, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  impl::polygamma_kernel_xpu(iter, n);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
