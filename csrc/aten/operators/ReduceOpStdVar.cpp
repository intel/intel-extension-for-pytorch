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

#include <oneapi/dpl/cmath>
namespace dpl = oneapi::dpl;

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
  DPCPP_BOTH WelfordData() : mean(0), m2(0), n(0), nf(0) {}
  DPCPP_DEVICE WelfordData(scalar_t mean, scalar_t m2, index_t n, combine_t nf)
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
  inline DPCPP_DEVICE acc_t
  reduce(acc_t acc, scalar_t data, int64_t idx) const {
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
  inline DPCPP_DEVICE acc_t combine(acc_t a, acc_t b) const {
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
  inline DPCPP_DEVICE res_t project(acc_t acc) const {
    auto mean = acc.mean;
    combine_t divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    auto ret = take_sqrt ? dpl::sqrt(var) : var;

    std::pair<scalar_t, scalar_t> results{(scalar_t)ret, (scalar_t)mean};
    return results;
  }
  inline DPCPP_DEVICE acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
  static inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) {
    return acc;
  }
  WelfordOps(index_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

template <typename scalar_t>
void std_var_kernel_impl(
    TensorIterator& iter,
    int64_t correction_opt,
    bool take_sqrt) {
  dpcpp_reduce_kernel<scalar_t, scalar_t, 2>(
      iter,
      WelfordOps<
          scalar_t,
          scalar_t,
          int32_t,
          float,
          std::pair<scalar_t, scalar_t>>{correction_opt, take_sqrt},
      WelfordData<scalar_t, int32_t, float>{});
}

template <>
void std_var_kernel_impl<at::Half>(
    TensorIterator& iter,
    int64_t correction_opt,
    bool take_sqrt) {
  dpcpp_reduce_kernel<at::Half, at::Half, 2>(
      iter,
      WelfordOps<
          at::Half,
          float,
          int32_t,
          float,
          std::pair<at::Half, at::Half>>{correction_opt, take_sqrt},
      WelfordData<float, int32_t, float>{});
}

static void std_var_kernel(
    TensorIterator& iter,
    int64_t correction_opt,
    bool take_sqrt) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "std",
      [&]() {
        std_var_kernel_impl<scalar_t>(iter, correction_opt, take_sqrt);
      });
}

Tensor& std_var_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    int64_t correction_opt,
    bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      "std and var only support floating-point dtypes");
  if (at::isComplexType(self.scalar_type())) {
    ScalarType dtype = c10::toRealValueType(get_dtype(result, self, {}, true));
    Tensor real_in = at::real(self).to(dtype);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    auto iter =
        make_reduction("std or var", real_out, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out.fill_(NAN);
    } else {
      std_var_kernel(iter, correction_opt, false);
    }
    Tensor imag_in = at::imag(self).to(dtype);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction("std or var", imag_out, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out.fill_(NAN);
    } else {
      std_var_kernel(iter, correction_opt, false);
    }
    at::add_out(result, real_out, imag_out);
    take_sqrt ? at::sqrt_out(result, result) : result;
  } else {
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("std or var", result, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result.fill_(NAN);
    } else {
      std_var_kernel(iter, correction_opt, take_sqrt);
    }
  }
  return result;
}

std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    IntArrayRef dim,
    int64_t correction_opt,
    bool keepdim,
    bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      fname,
      " only support floating-point dtypes");
  TORCH_CHECK(
      result1.scalar_type() == result2.scalar_type(),
      "provided by result1 dtype must match dtype of result2. Got ",
      toString(result1.scalar_type()),
      " and ",
      toString(result2.scalar_type()),
      ".");
  if (at::isComplexType(self.scalar_type())) {
    ScalarType dtype = c10::toRealValueType(get_dtype(result1, self, {}, true));
    Tensor real_in = at::real(self).to(dtype);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    auto iter = meta::make_reduction(
        fname, real_out_var, real_out_mean, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out_var.fill_(NAN);
      real_out_mean.fill_(NAN);
    } else {
      std_var_kernel(iter, correction_opt, false);
    }
    Tensor imag_in = at::imag(self).to(dtype);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    iter = meta::make_reduction(
        fname, imag_out_var, imag_out_mean, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out_var.fill_(NAN);
      imag_out_mean.fill_(NAN);
    } else {
      std_var_kernel(iter, correction_opt, false);
    }
    at::add_out(result1, real_out_var, imag_out_var);
    take_sqrt ? at::sqrt_out(result1, result1) : result1;
    at::add_out(
        result2,
        real_out_mean,
        at::mul(imag_out_mean, c10::complex<double>{0.0, 1.0}));
  } else {
    ScalarType dtype = get_dtype(result1, self, {}, true);
    auto iter = meta::make_reduction(
        fname, result1, result2, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result1.fill_(NAN);
      result2.fill_(NAN);
    } else {
      std_var_kernel(iter, correction_opt, take_sqrt);
    }
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

} // namespace AtenIpexTypeXPU
} // namespace at
