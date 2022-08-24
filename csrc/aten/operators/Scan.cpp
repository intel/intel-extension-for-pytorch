#include <ATen/ATen.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include "PSTLFunctions.h"
#include "Scan.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  if (self.dtype() == at::ScalarType::Bool) {
    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Bool, out.scalar_type(), "cumsum", [&]() {
          scan<INCLUSIVE_TYPE, bool, scalar_t>(
              out,
              self,
              dim,
              ScalarConvert<float, bool>::to(0.0),
              AddOp<bool>());
        });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, self.scalar_type(), "cumsum", [&]() {
          scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
              out,
              self,
              dim,
              ScalarConvert<float, scalar_t>::to(0.0),
              AddOp<scalar_t>());
        });
  }
  return out;
}

Tensor& cumprod_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "cumprod", [&]() {
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            out,
            self,
            dim,
            ScalarConvert<float, scalar_t>::to(1.0),
            MulOp<scalar_t>());
      });
  return out;
}

Tensor _logcumsumexp(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, at::MemoryFormat::Contiguous);
  return _logcumsumexp_out(self, dim, result);
}

Tensor& _logcumsumexp_out(const Tensor& self, int64_t dim, Tensor& result) {
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "logcumsumexp_out_dpcpp",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        scalar_t init = Numerics<scalar_t>::lower_bound();
        auto log_add_exp = [](const scalar_t x, const scalar_t y) -> scalar_t {
          // sycl::min returns first arg if one of the args is nan
          scalar_t min =
              Numerics<scalar_t>::isnan(y) ? y : Numerics<scalar_t>::min(x, y);
          // sycl::max returns first arg if one of the args is nan
          scalar_t max =
              Numerics<scalar_t>::isnan(y) ? y : Numerics<scalar_t>::max(x, y);
          if (min != max ||
              (!Numerics<accscalar_t>::isinf(static_cast<accscalar_t>(min)))) {
            // nan will be propagated here
            return Numerics<scalar_t>::log1p(
                       Numerics<scalar_t>::exp(min - max)) +
                max;
          } else {
            // special case to correctly handle infinite inputs
            return x;
          }
        };
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            result, self, wrap_dim, init, log_add_exp);
      });
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
