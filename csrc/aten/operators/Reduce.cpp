#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/Context.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <c10/core/ScalarType.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

#include "Loops.h"
#include "Reduce.h"

using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

using DimMask = TensorIterator::DimMask;

static DimMask make_dim_mask(IntArrayRef dims, int64_t ndim) {
  auto mask = DimMask();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      mask.set(maybe_wrap_dim(dim, ndim));
    }
  }
  return mask;
}

static void allocate_reduction_result(
    Tensor& result,
    const Tensor& self,
    DimMask mask,
    bool keepdim,
    ScalarType dtype) {
  auto shape = DimVector(self.sizes());
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  if (result.defined()) {
    result.resize_(shape);
  } else {
    result = at::empty(shape, self.options().dtype(dtype));
  }
}

static Tensor review_reduce_result(
    const Tensor& result,
    int ndim,
    DimMask mask,
    bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (int dim = 0; dim < ndim; dim++) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}

static TensorIterator make_reduction(
    const char* name,
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    ScalarType in_dtype,
    ScalarType out_dtype) {
  TORCH_CHECK(
      !result.defined() || result.scalar_type() == out_dtype,
      name,
      ": provided dtype must match dtype of result. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(out_dtype),
      ".");
  int64_t ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result, self, mask, keepdim, out_dtype);
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
#endif
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

static TensorIterator make_reduction(
    const char* name,
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    ScalarType out_dtype) {
  /* FIXME:
   * According to below comments, this check is only for mixed precision
  enabling,
   * which can handle the case while in_dtype == kHalf and out_dtype == kFloat.
   * When IPEX has this feature, we can enable below check, as well.
  // special case for type promotion in mixed precision, improves computational
  efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  const bool gpu_f16_to_f32 = (self.scalar_type() == kHalf && out_dtype ==
  kFloat); auto in_dtype = gpu_f16_to_f32 ? self.scalar_type() : out_dtype;
  return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
  */
  return make_reduction(name, result, self, dim, keepdim, out_dtype, out_dtype);
}

static TensorIterator make_reduction(
    const char* name,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  for (const Tensor* t : {&result1, &result2}) {
    const Tensor& result = *t;
    TORCH_CHECK(
        !result.defined() || result.scalar_type() == dtype,
        name,
        ": provided dtype must match dtype of result. Got ",
        toString(result.scalar_type()),
        " and ",
        toString(dtype),
        ".");
  }

  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result1, self, mask, keepdim, dtype);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  allocate_reduction_result(result2, self, mask, keepdim, dtype);
  auto viewed_result2 = review_reduce_result(result2, ndim, mask, keepdim);

#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names_for_reduction(result1, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(result2, self, dim, keepdim);
#endif

  /* FIXME:
   * This check is only for mixed precision enabling,
   * which can handle the case while in_dtype == kHalf and out_dtype == kFloat.
   * When IPEX has this feature, we can enable below check, as well.
   */
  if (self.scalar_type() == dtype
      /*|| (self.scalar_type() == kHalf && dtype == kFloat)*/) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }

  return TensorIterator::reduce_op(
      viewed_result1, viewed_result2, self.to(dtype));
}

static ScalarType get_dtype(
    Tensor& result,
    const Tensor& self,
    optional<ScalarType> dtype,
    bool promote_integers = false) {
  if (dtype.has_value()) {
    return dtype.value();
  } else if (result.defined()) {
    return result.scalar_type();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

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
  bool unbiased;
  bool take_sqrt;

 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline DPCPP_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
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
    combine_t divisor = unbiased ? (acc.nf - 1) : acc.nf;
    auto ret = (divisor > 0)
        ? (take_sqrt ? DPCPP::sqrt(acc.m2 / divisor) : (acc.m2 / divisor))
        : NAN;

    std::pair<scalar_t, scalar_t> results{(scalar_t)ret, (scalar_t)mean};
    return results;
  }
  inline DPCPP_DEVICE acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  WelfordOps(bool unbiased, bool take_sqrt)
      : unbiased(unbiased), take_sqrt(take_sqrt) {}
};

template <typename acc_t>
struct ReduceAddOps {
  ReduceAddOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return a + b;
  }
};

template <typename acc_t>
struct ReduceProdOps {
  ReduceProdOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return a * b;
  }
};

template <typename acc_t>
struct ReduceMinOps {
  ReduceMinOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return (Numerics<acc_t>::lt(a, b) || Numerics<acc_t>::isnan(a)) ? a : b;
  }
};

template <typename acc_t>
struct ReduceMaxOps {
  ReduceMaxOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return (Numerics<acc_t>::gt(a, b) || Numerics<acc_t>::isnan(a)) ? a : b;
  }
};

template <typename acc_t>
struct ReduceAndOps {
  ReduceAndOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return a && b;
  }
};

template <typename acc_t>
struct ReduceOrOps {
  ReduceOrOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return a || b;
  }
};

template <typename acc_t, typename factor_t>
struct ReduceMeanOps {
  factor_t factor;

  inline acc_t reduce(acc_t a, acc_t b) const {
    return a + b;
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return reduce(a, b);
  }

  inline acc_t project(acc_t a) const {
    return a * factor;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  ReduceMeanOps(factor_t factor) : factor(factor) {}
};

template <typename acc_t>
struct NormOps {
  acc_t norm;

  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + Numerics<acc_t>::pow(Numerics<acc_t>::fabs(data), norm);
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline acc_t project(acc_t a) const {
    return Numerics<acc_t>::pow(a, acc_t(1.0) / norm);
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  NormOps(acc_t norm) : norm(norm) {}
};

template <typename acc_t>
struct NormZeroOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + (data == acc_t(0) ? acc_t(0) : acc_t(1));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline acc_t project(acc_t a) const {
    return a;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename acc_t>
struct NormOneOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + Numerics<acc_t>::fabs(data);
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline acc_t project(acc_t a) const {
    return a;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename acc_t>
struct AbsMinOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return Numerics<acc_t>::min(acc, Numerics<acc_t>::fabs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return Numerics<acc_t>::min(a, b);
  }

  inline acc_t project(acc_t a) const {
    return a;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename acc_t>
struct AbsMaxOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return Numerics<acc_t>::max(acc, Numerics<acc_t>::fabs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return Numerics<acc_t>::max(a, b);
  }

  inline acc_t project(acc_t a) const {
    return a;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename scalar_t, typename acc_scalar_t, typename index_t>
struct MinMaxOps {
  using acc_t = std::pair<acc_scalar_t, acc_scalar_t>;
  inline acc_t reduce(acc_t acc, scalar_t data) const {
    return combine(acc, {data, data});
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    auto min_val = (Numerics<acc_scalar_t>::isnan(a.first) || a.first < b.first)
        ? a.first
        : b.first;
    auto max_val =
        (Numerics<acc_scalar_t>::isnan(a.second) || a.second > b.second)
        ? a.second
        : b.second;

    return {min_val, max_val};
  }

  inline acc_t project(acc_t acc) const {
    return acc;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  dpcpp_reduce_kernel<scalar_t, scalar_t, 2>(
      iter,
      WelfordOps<
          scalar_t,
          scalar_t,
          int32_t,
          float,
          std::pair<scalar_t, scalar_t>>{unbiased, take_sqrt},
      WelfordData<scalar_t, int32_t, float>{});
}

template <>
void std_var_kernel_impl<at::Half>(
    TensorIterator& iter,
    bool unbiased,
    bool take_sqrt) {
  dpcpp_reduce_kernel<at::Half, at::Half, 2>(
      iter,
      WelfordOps<
          at::Half,
          float,
          int32_t,
          float,
          std::pair<at::Half, at::Half>>{unbiased, take_sqrt},
      WelfordData<float, int32_t, float>{});
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<out_t>(ReduceAddOps<acc_t>()));
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void prod_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<out_t>(ReduceProdOps<acc_t>()), 1);
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  float factor = float(iter.num_output_elements()) / iter.numel();
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, ReduceMeanOps<acc_t, float>{factor});
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void min_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter,
      func_wrapper<scalar_t>(ReduceMinOps<scalar_t>()),
      Numerics<scalar_t>::upper_bound());
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void max_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter,
      func_wrapper<scalar_t>(ReduceMaxOps<scalar_t>()),
      Numerics<scalar_t>::lower_bound());
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
static void norm_kernel_impl(TensorIterator& iter, Scalar val) {
  float p = 0.0f;
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<acc_t>();
  } else {
    TORCH_CHECK(0, "norm_kernel_impl expects norm to be integer or float");
  }

  if (p == static_cast<float>(0)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<acc_t>(), 0);
  } else if (p == static_cast<float>(INFINITY)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(
        iter, AbsMaxOps<acc_t>(), std::numeric_limits<acc_t>::min());
  } else if (p == static_cast<float>(-INFINITY)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(
        iter, AbsMinOps<acc_t>(), std::numeric_limits<acc_t>::max());
  } else {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{acc_t(p)}, 0);
  }
}

static void std_var_kernel(
    TensorIterator& iter,
    bool unbiased,
    bool take_sqrt) {
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std", [&]() {
    std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
  });
}

static void sum_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "sum",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        sum_kernel_impl<scalar_t, accscalar_t>(iter);
      });
}

static void prod_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES(
      iter.dtype(), "prod", [&]() { prod_kernel_impl<scalar_t>(iter); });
}

static void mean_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mean",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        mean_kernel_impl<scalar_t, accscalar_t>(iter);
      });
}

static void min_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "min",
      [&]() { min_kernel_impl<scalar_t>(iter); });
}

static void max_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "max",
      [&]() { max_kernel_impl<scalar_t>(iter); });
}

static void norm_kernel(TensorIterator& iter, Scalar p) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    return norm_kernel_impl<at::Half, float, float>(iter, p);
  }
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "norm",
      [&]() { norm_kernel_impl<scalar_t>(iter, p); });
}

void and_kernel(TensorIterator& iter) {
  dpcpp_reduce_kernel<uint8_t, uint8_t>(
      iter, func_wrapper<uint8_t>(ReduceAndOps<uint8_t>()), true);
}

void or_kernel(TensorIterator& iter) {
  dpcpp_reduce_kernel<uint8_t, uint8_t>(
      iter, func_wrapper<uint8_t>(ReduceOrOps<uint8_t>()), false);
}

template <typename scalar_t>
void _min_max_values_kernel_dpcpp_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      std::pair<scalar_t, scalar_t>(
          at::numeric_limits<scalar_t>::upper_bound(),
          at::numeric_limits<scalar_t>::lower_bound()));
}

void aminmax_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "aminmax_elementwise_dpcpp",
      [&]() { _min_max_values_kernel_dpcpp_impl<scalar_t>(iter); });
}

} // namespace impl
using namespace impl;

Tensor& sum_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> opt_dtype) {
  ScalarType dtype = impl::get_dtype(result, self, opt_dtype, true);
  auto iter = impl::make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::sum_kernel(iter);
  }
  return result;
}

Tensor sum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::sum_out(result, self, dim, keepdim, dtype);
}

Tensor sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  return at::AtenIpexTypeXPU::sum(self, std::vector<int64_t>{}, false, dtype);
}

Tensor& prod_out_impl(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("prod", result, self, dims, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    impl::prod_kernel(iter);
  }
  return result;
}

Tensor& prod_out(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return at::AtenIpexTypeXPU::prod_out_impl(
      result, self, {dim}, keepdim, dtype);
}

Tensor prod(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::prod_out_impl(
      result, self, {dim}, keepdim, dtype);
}

Tensor prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::prod_out_impl(result, self, {}, false, dtype);
}

Tensor& mean_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  ScalarType scalarType =
      opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");

  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("mean", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    impl::mean_kernel(iter);
  }
  return result;
}

Tensor mean(const Tensor& self, optional<ScalarType> dtype) {
  return at::AtenIpexTypeXPU::mean(self, IntArrayRef{}, false, dtype);
}

Tensor mean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::mean_out(result, self, dim, keepdim, dtype);
}

Tensor min_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  ScalarType dtype = impl::get_dtype(result, self, c10::nullopt);
  auto iter = impl::make_reduction("min", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::min_kernel(iter);
  }
  return result;
}

Tensor min(const Tensor& self) {
  Tensor result;
  return at::AtenIpexTypeXPU::min_out(
      result, self, std::vector<int64_t>{}, false);
}

Tensor& amax_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "Illegal dtype for self, and out:",
      self.scalar_type(),
      result.scalar_type());
  auto iter = impl::make_reduction(
      "amax", result, self, dim, keepdim, self.scalar_type());
  TORCH_CHECK(iter.numel() > 0, "operation does not have an identity");
  impl::max_kernel(iter);
  return result;
}

Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::amax_out(result, self, dim, keepdim);
}

Tensor max(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::amax_out(
      result, self, std::vector<int64_t>{}, false);
}

// norm - static root entrances
static Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> opt_p,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  auto p = opt_p.value_or(2.0);
  ScalarType scalarType =
      opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");

  ScalarType dtype = impl::get_dtype(result, self, opt_dtype, true);
  auto iter = impl::make_reduction("norm", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::norm_kernel(iter, p);
  }
  return result;
}

static Tensor norm(
    const Tensor& self,
    optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::norm_out(
      result, self, p, dim, keepdim, opt_dtype);
}

static inline Tensor _norm(const Tensor& self, Scalar p) {
  if (self.is_sparse()) {
    return at::native_norm(self, p);
  } else {
    TORCH_CHECK(
        at::isFloatingType(self.scalar_type()) ||
            at::isComplexType(self.scalar_type()),
        "norm only supports floating-point dtypes");

    Tensor result;
    return at::AtenIpexTypeXPU::norm_out(
        result, self, p, IntArrayRef{}, false, c10::nullopt);
  }
}

// norm - customized outer entrances
Tensor& norm_out(
    Tensor& out,
    const Tensor& self,
    c10::optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  return at::AtenIpexTypeXPU::norm_out(
      out, self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor& norm_out(
    Tensor& out,
    const Tensor& self,
    c10::optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim) {
  return at::AtenIpexTypeXPU::norm_out(
      out, self, p, dim, keepdim, c10::nullopt);
}

Tensor norm(
    const Tensor& self,
    c10::optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  return at::AtenIpexTypeXPU::norm(
      self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, c10::optional<Scalar> p, ScalarType dtype) {
  return at::AtenIpexTypeXPU::norm(
      self, p, IntArrayRef{}, false, optional<ScalarType>(dtype));
}

Tensor norm(
    const Tensor& self,
    c10::optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim) {
  return at::AtenIpexTypeXPU::norm(self, p, dim, keepdim, c10::nullopt);
}

Tensor norm(const Tensor& self, Scalar p) {
  return at::AtenIpexTypeXPU::_norm(self, p);
}

inline Tensor& _all(Tensor& result, TensorIterator& iter) {
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    impl::and_kernel(iter);
  }

  return result;
}

Tensor all(const at::Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Byte ||
          self.scalar_type() == at::ScalarType::Bool,
      "all only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter =
      impl::make_reduction("all", result, self, {}, false, self.scalar_type());

  return at::AtenIpexTypeXPU::_all(result, iter);
}

Tensor& all_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Byte ||
          self.scalar_type() == at::ScalarType::Bool,
      "all only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    auto iter = impl::make_reduction(
        "all", result, self, dim, keepdim, self.scalar_type());
    return at::AtenIpexTypeXPU::_all(result, iter);
  }
}

Tensor all(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::all_out(result, self, dim, keepdim);
}

inline Tensor& _any(Tensor& result, TensorIterator& iter) {
  if (iter.numel() == 0) {
    result.fill_(0);
  } else {
    impl::or_kernel(iter);
  }

  return result;
}

Tensor any(const at::Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Byte ||
          self.scalar_type() == at::ScalarType::Bool,
      "any only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter =
      impl::make_reduction("any", result, self, {}, false, self.scalar_type());

  return at::AtenIpexTypeXPU::_any(result, iter);
}

Tensor& any_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Byte ||
          self.scalar_type() == at::ScalarType::Bool,
      "any only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    auto iter = impl::make_reduction(
        "any", result, self, dim, keepdim, self.scalar_type());
    return at::AtenIpexTypeXPU::_any(result, iter);
  }
}

Tensor any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::any_out(result, self, dim, keepdim);
}

class OpRenorm {};

Tensor& renorm_out(
    Tensor& out,
    const Tensor& self,
    Scalar p,
    int64_t dim,
    Scalar maxnorm) {
  TORCH_CHECK(!self.is_sparse(), "renorm(sycl_sparse) is not supported.");
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "invalid dimension dim=", dim);
  TORCH_CHECK(p.toFloat() > 0, "non-positive-norm not supported");
  TORCH_CHECK(self.dim() > 1, "need at least 2 dimensions, got ", self.dim());

  auto norm_vec_sz = self.size(dim);
  Tensor norm = at::empty(norm_vec_sz, self.options().dtype(kFloat));
  at::AtenIpexTypeXPU::norm_out(
      norm,
      self.transpose(0, dim).reshape({norm_vec_sz, -1}),
      p,
      IntArrayRef(1),
      false,
      c10::nullopt);

  auto iter = TensorIteratorConfig()
                  .add_output(norm)
                  .add_input(norm)
                  .set_check_mem_overlap(true)
                  .build();
  float maxnorm_ = maxnorm.toFloat();
  dpcpp_kernel_for_tensor_iter<OpRenorm>(iter, [=](float norm) -> float {
    if (norm > maxnorm_)
      return maxnorm_ / (norm + 1e-7);
    return 1;
  });

  std::vector<int64_t> sizes_;
  sizes_.push_back(norm_vec_sz);
  size_t tailing_dims = self.dim() - (dim + 1);
  for (size_t dimension = 0; dimension < tailing_dims; ++dimension) {
    sizes_.push_back(1);
  }

  return at::AtenIpexTypeXPU::mul_out(
      out, self, norm.contiguous().view(sizes_));
}

Tensor renorm(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  TORCH_CHECK(!self.is_sparse(), "renorm(sycl_sparse) is not supported.");

  Tensor result;
  result = at::empty(self.sizes(), self.options());
  at::AtenIpexTypeXPU::renorm_out(result, self, p, dim, maxnorm);
  return result;
}

Tensor& renorm_(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::AtenIpexTypeXPU::renorm_out(self, self, p, dim, maxnorm);
}

Tensor& std_var_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      "std and var only support floating-point dtypes");
  if (at::isComplexType(self.scalar_type())) {
    ScalarType dtype = c10::toValueType(get_dtype(result, self, {}, true));
    Tensor real_in = at::real(self).to(dtype);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    auto iter =
        make_reduction("std or var", real_out, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out.fill_(NAN);
    } else {
      std_var_kernel(iter, unbiased, false);
    }
    Tensor imag_in = at::imag(self).to(dtype);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction("std or var", imag_out, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out.fill_(NAN);
    } else {
      std_var_kernel(iter, unbiased, false);
    }
    at::add_out(result, real_out, imag_out);
    take_sqrt ? at::sqrt_out(result, result) : result;
  } else {
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("std or var", result, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result.fill_(NAN);
    } else {
      std_var_kernel(iter, unbiased, take_sqrt);
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
    bool unbiased,
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
    ScalarType dtype = c10::toValueType(get_dtype(result1, self, {}, true));
    Tensor real_in = at::real(self).to(dtype);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    auto iter = make_reduction(
        fname, real_out_var, real_out_mean, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out_var.fill_(NAN);
      real_out_mean.fill_(NAN);
    } else {
      std_var_kernel(iter, unbiased, false);
    }
    Tensor imag_in = at::imag(self).to(dtype);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction(
        fname, imag_out_var, imag_out_mean, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out_var.fill_(NAN);
      imag_out_mean.fill_(NAN);
    } else {
      std_var_kernel(iter, unbiased, false);
    }
    at::add_out(result1, real_out_var, imag_out_var);
    take_sqrt ? at::sqrt_out(result1, result1) : result1;
    at::add_out(
        result2,
        real_out_mean,
        at::mul(imag_out_mean, c10::complex<double>{0.0, 1.0}));
  } else {
    ScalarType dtype = get_dtype(result1, self, {}, true);
    auto iter =
        make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result1.fill_(NAN);
      result2.fill_(NAN);
    } else {
      std_var_kernel(iter, unbiased, take_sqrt);
    }
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

Tensor& argmax_out(
    Tensor& result,
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function argmax on a "
      "tensor with no elements because the operation does not have an "
      "identity");
  Tensor in;
  if (dim) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }

  Tensor ignored = at::empty({0}, self.options());
  return std::get<1>(
      at::max_out(ignored, result, in, dim.value_or(0), keepdim));
}

Tensor argmax(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return argmax_out(result, self, dim, keepdims);
}

Tensor& argmin_out(
    Tensor& result,
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function argmin on a "
      "tensor with no elements because the operation does not have an "
      "identity");
  Tensor in;
  if (dim) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }

  Tensor ignored = at::empty({0}, self.options());
  return std::get<1>(
      at::min_out(ignored, result, in, dim.value_or(0), keepdim));
}

Tensor argmin(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return argmin_out(result, self, dim, keepdims);
}

void aminmax_out(Tensor& min_result, Tensor& max_result, const Tensor& self) {
  auto iter = impl::make_reduction(
      "aminmax",
      min_result,
      max_result,
      self,
      std::vector<int64_t>{},
      false,
      self.scalar_type()); // TensorIterator::binary_op(min_result, max_result,
                           // self);
  impl::aminmax_kernel(iter);
}

std::tuple<Tensor, Tensor> _aminmax(const Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor min_result = at::empty_like(self);
  Tensor max_result = at::empty_like(self);
  at::AtenIpexTypeXPU::aminmax_out(min_result, max_result, self);
  return std::tuple<Tensor&, Tensor&>(min_result, max_result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
