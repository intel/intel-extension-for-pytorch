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
#include "ReduceOpStdVar.h"
#include "ReduceOpsUtils.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;
  WelfordData() : mean(0), m2(0), n(0), nf(0) {}
  WelfordData(scalar_t mean, scalar_t m2, index_t n, combine_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <
    typename scalar_t,
    typename acc_scalar_t,
    typename index_t,
    typename combine_t,
    typename res_t>
struct WelfordOps {
  index_t correction;
  ;
  bool take_sqrt;

 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
        new_mean,
        acc.m2 + delta * new_delta,
        acc.n + 1,
        combine_t(acc.n + 1),
    };
  }
  inline acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
        a.mean + delta * nb_over_n,
        a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
        -1,
        new_count};
  }
  inline res_t project(acc_t acc) const {
    auto mean = acc.mean;
    combine_t divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    auto ret = take_sqrt ? std::sqrt(var) : var;

    std::pair<scalar_t, scalar_t> results{(scalar_t)ret, (scalar_t)mean};
    return results;
  }
  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
  static inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) {
    return acc;
  }
  WelfordOps(index_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

template <typename scalar_t, typename out_t = scalar_t>
void std_var_kernel_impl(
    TensorIterator& iter,
    int64_t correction,
    bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  dpcpp_reduce_kernel<scalar_t, out_t, 2>(
      iter,
      WelfordOps<scalar_t, scalar_t, int32_t, float, std::pair<out_t, out_t>>{
          correction, take_sqrt},
      WelfordData<scalar_t, int32_t, float>{});
}

static void std_var_kernel(
    TensorIterator& iter,
    int64_t correction,
    bool take_sqrt) {
  const auto input_dtype = iter.input_dtype();
  if (input_dtype == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::Half, float>(iter, correction, take_sqrt);
  } else if (input_dtype == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::BFloat16, float>(iter, correction, take_sqrt);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "std",
        [&]() { std_var_kernel_impl<scalar_t>(iter, correction, take_sqrt); });
  }
}

Tensor& std_var_out(
    Tensor& result,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(
      self.device().is_cpu() || self.device().is_xpu(),
      "std and var only supports tensors on a CPU or XPU device, but got: ",
      self.device().type());
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "std and var only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      "std and var only support floating point and complex dtypes");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate variance of real and imaginary components
    // separately then add to get overall variance.
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result, {}));
    Tensor real_in = at::real(self);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        real_out,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        imag_out,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result, real_out, imag_out);
    if (take_sqrt) {
      at::sqrt_out(result, result);
    }
    return result;
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result, {});
  auto iter = make_reduction("std or var", result, self, dim, keepdim, dtype);
  TORCH_CHECK(
      at::canCast(self.scalar_type(), result.scalar_type()),
      "result type ",
      self.scalar_type(),
      " can't be cast to the "
      "desired output type ",
      result.scalar_type());

  if (iter.numel() == 0) {
    // Trivial reduction
    result.fill_(std::numeric_limits<double>::quiet_NaN());
    return result;
  } else if (
      result.numel() == 1 && iter.device_type() == kCPU &&
      iter.common_dtype() != kBFloat16 && iter.common_dtype() != kHalf) {
    // NOTE: CPU performance significantly regressed when attempting to port to
    // ATen,
    //   so all-reduce has a custom implementation.
    //   See https://github.com/pytorch/pytorch/pull/43858.

    // result.fill_(std_var_all_cpu(self, correction, take_sqrt));
  } else {
    std_var_kernel(iter, correction, take_sqrt);
  }
  return result;
}

std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(
      self.device().is_cpu() || self.is_xpu(),
      fname,
      " only supports tensors on a CPU or XPU device, got: ",
      self.device().type());
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      fname,
      " only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      fname,
      " only support floating point and complex dtypes");
  TORCH_CHECK(
      result1.scalar_type() == c10::toRealValueType(result2.scalar_type()),
      fname,
      " expected result1 to be real and match the precision of result2. Got ",
      result1.scalar_type(),
      " and ",
      result2.scalar_type(),
      ".");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate for real and imaginary components separately then
    // combine as: variance = var_real + var_imag mean = mean_real + j *
    // mean_imag
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result1, {}));
    Tensor real_in = at::real(self);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        real_out_var,
        real_out_mean,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        imag_out_var,
        imag_out_mean,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result1, real_out_var, imag_out_var);
    if (take_sqrt) {
      at::sqrt_out(result1, result1);
    }
    at::complex_out(result2, real_out_mean, imag_out_mean);
    return std::tuple<Tensor&, Tensor&>(result1, result2);
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result1, {});
  auto iter =
      make_reduction(fname, result1, result2, self, dim, keepdim, dtype);

  if (iter.numel() == 0) {
    // Trivial reduction
    result1.fill_(std::numeric_limits<double>::quiet_NaN());
    result2.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    std_var_kernel(iter, correction, take_sqrt);
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

} // namespace AtenIpexTypeXPU
} // namespace at
