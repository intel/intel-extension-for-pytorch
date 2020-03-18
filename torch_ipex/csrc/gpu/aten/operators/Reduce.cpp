#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <utils/Numerics.h>

#include "Loops.h"
#include "Reduce.h"

#include <iostream>

using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
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

static void allocate_reduction_result(Tensor &result, const Tensor &self,
                                      DimMask mask, bool keepdim,
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

static Tensor review_reduce_result(const Tensor &result, int ndim, DimMask mask,
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

static TensorIterator make_reduction(const char *name, Tensor &result,
                                     const Tensor &self, IntArrayRef dim,
                                     bool keepdim, ScalarType in_dtype,
                                     ScalarType out_dtype) {
  // check that result type and dtype match if provided
  TORCH_CHECK(!result.defined() || result.scalar_type() == out_dtype, name,
              ": provided dtype must match dtype of result. Got ",
              toString(result.scalar_type()), " and ", toString(out_dtype),
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

static TensorIterator make_reduction(const char *name, Tensor &result,
                                     const Tensor &self, IntArrayRef dim,
                                     bool keepdim, ScalarType out_dtype) {
  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  const bool gpu_f16_to_f32 =
      (self.is_cuda() && self.scalar_type() == kHalf && out_dtype == kFloat);
  auto in_dtype = gpu_f16_to_f32 ? self.scalar_type() : out_dtype;
  return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
}

static TensorIterator make_reduction(const char *name, Tensor &result1,
                                     Tensor &result2, const Tensor &self,
                                     IntArrayRef dim, bool keepdim,
                                     ScalarType dtype) {
  // check that result type and dtype match if provided
  for (const Tensor *t : {&result1, &result2}) {
    const Tensor &result = *t;
    TORCH_CHECK(!result.defined() || result.scalar_type() == dtype, name,
                ": provided dtype must match dtype of result. Got ",
                toString(result.scalar_type()), " and ", toString(dtype), ".");
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

  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // We don't generalize this to common mismatched input/output types to avoid
  // cross
  // product of templated kernel launches.
  if (self.scalar_type() == dtype ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  return TensorIterator::reduce_op(viewed_result1, viewed_result2,
                                   self.to(dtype));
}

static ScalarType get_dtype(Tensor &result, const Tensor &self,
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

template <typename scalar_t, typename acc_scalar_t, typename index_t,
          typename combine_t, typename res_t>
struct WelfordOps {
  bool unbiased;
  bool take_sqrt;

public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline DPCPP_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
        new_mean, acc.m2 + delta * new_delta, acc.n + 1,
        combine_t(acc.n + 1), // accumulate for combine_t uses index_t
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
    return {a.mean + delta * nb_over_n,
            a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
            // setting acc.n as -1 since acc.n might not be able to represent
            // the count
            // correctly within its range, setting it to -1 to avoid confusion
            -1, new_count};
  }
  inline DPCPP_DEVICE res_t project(acc_t acc) const {
    auto mean = acc.mean;
    combine_t divisor = unbiased ? (acc.nf - 1) : acc.nf;
    auto ret = (divisor > 0) ? (take_sqrt ? DPCPP::sqrt(acc.m2 / divisor)
                                          : (acc.m2 / divisor))
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

template <typename acc_t> struct add_kernel {
  add_kernel() {}
  acc_t operator()(acc_t a, acc_t b) const { return a + b; }
};

template <typename acc_t> struct min_kernel {
  min_kernel() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return (Numerics<acc_t>::lt(a, b) || Numerics<acc_t>::isnan(a)) ? a : b;
  }
};

template <typename acc_t> struct max_kernel {
  max_kernel() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return (Numerics<acc_t>::gt(a, b) || Numerics<acc_t>::isnan(a)) ? a : b;
  }
};

template <typename acc_t> struct and_kernel {
  and_kernel() {}
  acc_t operator()(acc_t a, acc_t b) const { return a && b; }
};

template <typename acc_t> struct or_kernel {
  or_kernel() {}
  acc_t operator()(acc_t a, acc_t b) const { return a || b; }
};

template <typename acc_t, typename factor_t> struct MeanOps {
  factor_t factor;

  inline acc_t reduce(acc_t a, acc_t b) const { return a + b; }

  inline acc_t combine(acc_t a, acc_t b) const { return reduce(a, b); }

  inline acc_t project(acc_t a) const { return a * factor; }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  MeanOps(factor_t factor) : factor(factor) {}
};

template <typename acc_t> struct NormOps {
  acc_t norm;

  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + DPCPP::pow(DPCPP::fabs(data), norm);
  }

  inline acc_t combine(acc_t a, acc_t b) const { return a + b; }

  inline acc_t project(acc_t a) const {
    return DPCPP::pow(a, acc_t(1.0) / norm);
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  NormOps(acc_t norm) : norm(norm) {}
};

template <typename acc_t> struct NormZeroOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + (data == acc_t(0) ? acc_t(0) : acc_t(1));
  }

  inline acc_t combine(acc_t a, acc_t b) const { return a + b; }

  inline acc_t project(acc_t a) const { return a; }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename acc_t> struct NormOneOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + DPCPP::fabs(data);
  }

  inline acc_t combine(acc_t a, acc_t b) const { return a + b; }

  inline acc_t project(acc_t a) const { return a; }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename acc_t> struct AbsMinOps {

  inline acc_t reduce(acc_t acc, acc_t data) const {
    return DPCPP::min(acc, DPCPP::fabs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const { return DPCPP::min(a, b); }

  inline acc_t project(acc_t a) const { return a; }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename acc_t> struct AbsMaxOps {

  inline acc_t reduce(acc_t acc, acc_t data) const {
    return DPCPP::max(acc, DPCPP::fabs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const { return DPCPP::max(a, b); }

  inline acc_t project(acc_t a) const { return a; }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename scalar_t>
void std_var_kernel_impl(TensorIterator &iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  dpcpp_reduce_kernel<scalar_t, scalar_t, 2>(
      iter, WelfordOps<scalar_t, scalar_t, int32_t, float,
                       std::pair<scalar_t, scalar_t>>{unbiased, take_sqrt},
      WelfordData<scalar_t, int32_t, float>{});
}

template <>
void std_var_kernel_impl<at::Half>(TensorIterator &iter, bool unbiased,
                                   bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  dpcpp_reduce_kernel<at::Half, at::Half, 2>(
      iter, WelfordOps<at::Half, float, int32_t, float,
                       std::pair<at::Half, at::Half>>{unbiased, take_sqrt},
      WelfordData<float, int32_t, float>{});
}

template <typename scalar_t, typename acc_t = scalar_t,
          typename out_t = scalar_t>
void sum_kernel_impl(TensorIterator &iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<out_t>(add_kernel<acc_t>()));
}

template <typename scalar_t, typename acc_t = scalar_t,
          typename out_t = scalar_t>
void mean_kernel_impl(TensorIterator &iter) {
  float factor = float(iter.num_output_elements()) / iter.numel();
  dpcpp_reduce_kernel<scalar_t, out_t>(iter, MeanOps<acc_t, float>{factor});
}

template <typename scalar_t, typename acc_t = scalar_t,
          typename out_t = scalar_t>
void min_kernel_impl(TensorIterator &iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<scalar_t>(min_kernel<scalar_t>()),
      Numerics<scalar_t>::upper_bound());
}

template <typename scalar_t, typename acc_t = scalar_t,
          typename out_t = scalar_t>
void max_kernel_impl(TensorIterator &iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<scalar_t>(max_kernel<scalar_t>()),
      Numerics<scalar_t>::lower_bound());
}

template <typename scalar_t, typename acc_t = scalar_t,
          typename out_t = scalar_t>
static void norm_kernel_impl(TensorIterator &iter, Scalar val) {
  float p;
  if (val.isIntegral()) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<acc_t>();
  } else {
    AT_ERROR("norm_kernel_impl expects norm to be integer or float");
  }

  if (p == static_cast<float>(0)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<acc_t>(), 0);
  } else if (p == static_cast<float>(INFINITY)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<acc_t>(),
                                         std::numeric_limits<acc_t>::min());
  } else if (p == static_cast<float>(-INFINITY)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<acc_t>(),
                                         std::numeric_limits<acc_t>::max());
  } else {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{acc_t(p)}, 0);
  }
}

static void std_var_kernel_dpcpp(TensorIterator &iter, bool unbiased,
                                 bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std", [&]() {
    std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
  });
}

static void sum_kernel_dpcpp(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "sum",
                        [&]() { sum_kernel_impl<scalar_t>(iter); });
}

static void prod_kernel_dpcpp(TensorIterator &iter) {
  AT_ERROR("prod_kernel_dpcpp not implemented yet!");
}

static void mean_kernel_dpcpp(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mean",
                        [&]() { mean_kernel_impl<scalar_t>(iter); });
}

static void min_kernel_dpcpp(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "min",
                        [&]() { min_kernel_impl<scalar_t>(iter); });
}

static void max_kernel_dpcpp(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "max",
                        [&]() { max_kernel_impl<scalar_t>(iter); });
}

static void norm_kernel_dpcpp(TensorIterator &iter, Scalar p) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_impl<at::Half, float, float>(iter, p);
  }
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm",
                             [&]() { norm_kernel_impl<scalar_t>(iter, p); });
}

void and_kernel_dpcpp(TensorIterator &iter) {
  dpcpp_reduce_kernel<uint8_t, uint8_t>(
      iter, func_wrapper<uint8_t>(and_kernel<uint8_t>()), true);
}

void or_kernel_dpcpp(TensorIterator &iter) {
  dpcpp_reduce_kernel<uint8_t, uint8_t>(
      iter, func_wrapper<uint8_t>(or_kernel<uint8_t>()), false);
}

} // namespace impl

Tensor &sum_out(Tensor &result, const Tensor &self, IntArrayRef dim,
                bool keepdim, c10::optional<at::ScalarType> opt_dtype) {
  ScalarType dtype = impl::get_dtype(result, self, opt_dtype, true);
  auto iter = impl::make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::sum_kernel_dpcpp(iter);
  }
  return result;
}

Tensor sum(const Tensor &self, IntArrayRef dim, bool keepdim,
           c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeDPCPP::sum_out(result, self, dim, keepdim, dtype);
}

Tensor sum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::AtenIpexTypeDPCPP::sum(self, std::vector<int64_t>{}, false, dtype);
}

Tensor min_out(Tensor &result, const Tensor &self, IntArrayRef dim,
               bool keepdim) {
  ScalarType dtype = impl::get_dtype(result, self, c10::nullopt);
  auto iter = impl::make_reduction("min", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::min_kernel_dpcpp(iter);
  }
  return result;
}

Tensor min(const Tensor &self) {
  Tensor result;
  return at::AtenIpexTypeDPCPP::min_out(result, self, std::vector<int64_t>{},
                                        false);
}

Tensor max_out(Tensor &result, const Tensor &self, IntArrayRef dim,
               bool keepdim) {
  ScalarType dtype = impl::get_dtype(result, self, c10::nullopt);
  auto iter = impl::make_reduction("max", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::max_kernel_dpcpp(iter);
  }
  return result;
}

Tensor max(const Tensor &self) {
  Tensor result;
  return at::AtenIpexTypeDPCPP::max_out(result, self, std::vector<int64_t>{},
                                        false);
}

static Tensor &norm_out(Tensor &result, const Tensor &self,
                        optional<Scalar> opt_p, IntArrayRef dim, bool keepdim,
                        optional<ScalarType> opt_dtype) {
  auto p = opt_p.value_or(2.0);
  ScalarType scalarType =
      opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(at::isFloatingType(scalarType) || at::isComplexType(scalarType),
              "Can only calculate the mean of floating types. Got ",
              toString(scalarType), " instead.");

  ScalarType dtype = impl::get_dtype(result, self, opt_dtype, true);
  auto iter = impl::make_reduction("norm", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    impl::norm_kernel_dpcpp(iter, p);
  }
  return result;
}

static Tensor norm(const Tensor &self, optional<Scalar> p, IntArrayRef dim,
                   bool keepdim, optional<ScalarType> opt_dtype) {
  Tensor result;
  return at::AtenIpexTypeDPCPP::norm_out(result, self, p, dim, keepdim,
                                         opt_dtype);
}

static inline Tensor _norm(const Tensor &self, Scalar p) {
  if (self.is_sparse()) {
    return at::native_norm(self, p);
  } else {
    TORCH_CHECK(at::isFloatingType(self.scalar_type()) ||
                    at::isComplexType(self.scalar_type()),
                "norm only supports floating-point dtypes");

    Tensor result;
    return at::AtenIpexTypeDPCPP::norm_out(result, self, p, IntArrayRef{},
                                           false, c10::nullopt);
  }
}

Tensor &norm_out(Tensor &out, const Tensor &self, c10::optional<Scalar> p,
                 IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::AtenIpexTypeDPCPP::norm_out(out, self, p, dim, keepdim, dtype);
}

Tensor &norm_out(Tensor &out, const Tensor &self, c10::optional<Scalar> p,
                 IntArrayRef dim, bool keepdim) {
  return at::AtenIpexTypeDPCPP::norm_out(out, self, p, dim, keepdim,
                                         c10::nullopt);
}

Tensor norm(const Tensor &self, c10::optional<Scalar> p, IntArrayRef dim,
            bool keepdim, ScalarType dtype) {
  return at::AtenIpexTypeDPCPP::norm(self, p, dim, keepdim,
                                     optional<ScalarType>(dtype));
}

Tensor norm(const Tensor &self, c10::optional<Scalar> p, ScalarType dtype) {
  return at::AtenIpexTypeDPCPP::norm(self, p, IntArrayRef{}, false,
                                     optional<ScalarType>(dtype));
}

Tensor norm(const Tensor &self, c10::optional<Scalar> p, IntArrayRef dim,
            bool keepdim) {
  return at::AtenIpexTypeDPCPP::norm(self, p, dim, keepdim, c10::nullopt);
}

Tensor norm(const Tensor &self, Scalar p) {
  return at::AtenIpexTypeDPCPP::_norm(self, p);
}

inline Tensor &_all(Tensor &result, TensorIterator &iter) {
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    impl::and_kernel_dpcpp(iter);
  }

  return result;
}

Tensor all(const at::Tensor &self) {
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte ||
                  self.scalar_type() == at::ScalarType::Bool,
              "all only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter =
      impl::make_reduction("all", result, self, {}, false, self.scalar_type());

  return at::AtenIpexTypeDPCPP::_all(result, iter);
}

Tensor &all_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte ||
                  self.scalar_type() == at::ScalarType::Bool,
              "all only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    auto iter = impl::make_reduction("all", result, self, dim, keepdim,
                                     self.scalar_type());
    return at::AtenIpexTypeDPCPP::_all(result, iter);
  }
}

Tensor all(const Tensor &self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::all_out(result, self, dim, keepdim);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
