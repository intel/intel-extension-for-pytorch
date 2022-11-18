#include <ATen/Context.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>

#include <c10/core/ScalarType.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Reduce.h"
#include "ReduceOpsUtils.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

template <typename acc_t, typename factor_t>
struct ReduceMeanOps {
  factor_t factor;

  inline acc_t reduce(acc_t a, acc_t b, int64_t idx) const {
    return a + b;
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline acc_t project(acc_t a) const {
    return a * factor;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }

  ReduceMeanOps(factor_t factor) : factor(factor) {}
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  //  returns acc_t for all non-complex dtypes and returns T for c10::complex<T>
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor =
      static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, ReduceMeanOps<acc_t, factor_t>{factor});
}

static void mean_kernel(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if (iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(
        iter.dtype(), "mean", [&]() { mean_kernel_impl<scalar_t>(iter); });
  }
}

Tensor& mean_out(
    const Tensor& self,
    c10::OptionalArrayRef<long> opt_dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype,
    Tensor& result) {
  ScalarType scalarType =
      opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");

  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto dim = opt_dim.value_or(IntArrayRef{});
  auto iter = meta::make_reduction("mean", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    mean_kernel(iter);
  }
  return result;
}

Tensor mean(
    const Tensor& self,
    c10::OptionalArrayRef<long> opt_dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::mean_out(self, opt_dim, keepdim, dtype, result);
}

Tensor mean(const Tensor& self, optional<ScalarType> dtype) {
  return at::AtenIpexTypeXPU::mean(
      self, OptionalIntArrayRef{IntArrayRef{}}, false, dtype);
}

} // namespace AtenIpexTypeXPU
} // namespace at
