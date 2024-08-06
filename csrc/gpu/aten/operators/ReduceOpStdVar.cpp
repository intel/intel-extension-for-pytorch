#include <ATen/Context.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <c10/core/ScalarType.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/XPUPair.h"

#include "Reduce.h"
#include "ReduceOpStdVar.h"
#include "ReduceOpsUtils.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;
  WelfordData() : mean(0), m2(0), n(0), nf(0) {}
  WelfordData(scalar_t mean, scalar_t m2, index_t n, scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <
    typename scalar_t,
    typename acc_scalar_t,
    typename index_t,
    typename res_t>
struct WelfordOps {
  index_t correction;
  bool take_sqrt;

 public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  inline acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // need nf for use in combine where int32 may overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {
        new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
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
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
        a.mean + delta * nb_over_n,
        a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
        -1,
        new_count};
  }
  inline res_t project(acc_t acc) const {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    auto ret = take_sqrt ? std::sqrt(var) : var;

    at::AtenIpexTypeXPU::pair<scalar_t, scalar_t> results{
        (scalar_t)ret, (scalar_t)mean};
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
    double correction_opt,
    bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  using accscalar_t = AtenIpexTypeXPU::acc_type<scalar_t>;
  using ops_t = WelfordOps<
      scalar_t,
      accscalar_t,
      int32_t,
      at::AtenIpexTypeXPU::pair<out_t, out_t>>;
  ops_t ops(static_cast<accscalar_t>(correction_opt), take_sqrt);
  dpcpp_reduce_kernel<scalar_t, out_t, 2>(iter, ops, typename ops_t::acc_t{});
}

static void std_var_kernel(
    TensorIterator& iter,
    double correction_opt,
    bool take_sqrt) {
  const auto input_dtype = iter.input_dtype();
  if (input_dtype == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::Half, float>(iter, correction_opt, take_sqrt);
  } else if (input_dtype == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::BFloat16, float>(iter, correction_opt, take_sqrt);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "std_dpcpp",
        [&]() {
          std_var_kernel_impl<scalar_t>(iter, correction_opt, take_sqrt);
        });
  }
}

static double std_var_all_cpu(
    const Tensor& self,
    double correction,
    bool take_sqrt) {
  const auto dtype = self.scalar_type();
  TORCH_CHECK(
      dtype == kDouble || dtype == kFloat,
      "std_var_all: Unsupported dtype ",
      dtype);

  auto mean = self.mean().item<double>();
  auto iter = TensorIteratorConfig().add_input(self).build();

  auto reduction = [&](int64_t begin, int64_t end, double thread_sum) {
    IPEX_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "std_var_all_cpu", [&] {
      iter.serial_for_each(
          [&](char** data,
              const int64_t* strides,
              int64_t size0,
              int64_t size1) {
            const double local_mean = mean;
            const int64_t inner_stride = strides[0];
            const int64_t outer_stride = strides[1];

            double local_sum = 0.0;
            for (const auto i : c10::irange(size1)) {
              const char* row_ptr = data[0] + outer_stride * i;
              for (const auto j : c10::irange(size0)) {
                const auto ptr = reinterpret_cast<const scalar_t*>(
                    row_ptr + inner_stride * j);
                auto dx = (static_cast<double>(*ptr) - local_mean);
                local_sum += dx * dx;
              }
            }
            thread_sum += local_sum;
          },
          {begin, end});
    });

    return thread_sum;
  };

  // ((x - mean)**2).sum()
  const double sum_dx2 = at::parallel_reduce(
      0, iter.numel(), at::internal::GRAIN_SIZE, 0.0, reduction, std::plus<>{});

  const auto var = [&]() __ubsan_ignore_float_divide_by_zero__ {
    return sum_dx2 / std::max(0.0, self.numel() - correction);
  }();
  const auto result = take_sqrt ? std::sqrt(var) : var;

  if (dtype == kFloat) {
    // Convert to infinity if out of range for a float.
    // Doing it now prevents checked_convert failing later
    return static_cast<float>(result);
  }
  return result;
}

Tensor& std_var_out(
    const char* fname,
    at::Tensor& result,
    const at::Tensor& self,
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
        fname,
        real_out,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        fname,
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
  auto iter = make_reduction(fname, result, self, dim, keepdim, dtype);
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
    result.fill_(std_var_all_cpu(self, correction, take_sqrt));
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
