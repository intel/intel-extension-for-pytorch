#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <runtime/Utils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include "comm/Numerics.h"
#include "comm/ATDispatch.h"

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



} // namespace AtenIpexTypeXPU
} // namespace at
