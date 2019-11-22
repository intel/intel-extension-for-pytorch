#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/dpcpp/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/dpcpp/Reduce.h>

#include <iostream>

namespace at { namespace native {

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;
  DP_BOTH WelfordData() : mean(0), m2(0), n(0), nf(0)  {}
  DP_DEVICE WelfordData(scalar_t mean, scalar_t m2, index_t n, combine_t nf) : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t, typename res_t>
struct WelfordOps {
  bool unbiased;
  bool take_sqrt;
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline DP_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1,
      combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline DP_DEVICE acc_t combine(acc_t a, acc_t b) const {
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
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline DP_DEVICE res_t project(acc_t acc) const {
    auto mean = acc.mean;
    combine_t divisor = unbiased ? (acc.nf - 1) : acc.nf;
    auto ret = (divisor > 0) ?
      (take_sqrt ? DP::sqrt(acc.m2 / divisor) : (acc.m2 / divisor))
      : NAN;

    std::pair<scalar_t, scalar_t> results{(scalar_t) ret, (scalar_t) mean};
    return results;
  }
  inline DP_DEVICE acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  WelfordOps(bool unbiased, bool take_sqrt)
    : unbiased(unbiased), take_sqrt(take_sqrt) {
  }
};

template <typename acc_t>
struct add_kernel{
  add_kernel() {}
  acc_t operator ()(acc_t a, acc_t b) const {
    return a + b;
  }
};

template <typename acc_t>
struct and_kernel{
  and_kernel() {}
  acc_t operator ()(acc_t a, acc_t b) const {
    return a && b;
  }
};

template <typename acc_t>
struct or_kernel{
  or_kernel() {}
  acc_t operator ()(acc_t a, acc_t b) const {
    return a || b;
  }
};

template <typename acc_t, typename factor_t>
struct MeanOps {
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

  MeanOps(factor_t factor): factor(factor) {
  }
};

template <typename acc_t>
struct NormOps {
  acc_t norm;

  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + cl::sycl::pow(cl::sycl::fabs(data), norm);
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline acc_t project(acc_t a) const {
    return cl::sycl::pow(a, acc_t(1.0)/norm);
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }

  NormOps(acc_t norm): norm(norm) {
  }
};

template <typename acc_t>
struct NormZeroOps {
  inline acc_t reduce(acc_t acc, acc_t data) const {
    return acc + (data==acc_t(0) ? acc_t(0) : acc_t(1));
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
    return acc + cl::sycl::fabs(data);
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
    return cl::sycl::min(acc, cl::sycl::fabs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return cl::sycl::min(a, b);
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
    return cl::sycl::max(acc, cl::sycl::fabs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return cl::sycl::max(a, b);
  }

  inline acc_t project(acc_t a) const {
    return a;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
};

template <typename scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  sycl_reduce_kernel<scalar_t, scalar_t, 2>(iter, WelfordOps<scalar_t, scalar_t, int32_t, float, std::pair<scalar_t, scalar_t>> { unbiased, take_sqrt }, WelfordData<scalar_t, int32_t, float> {});
}

template <>
void std_var_kernel_impl<at::Half>(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  sycl_reduce_kernel<at::Half, at::Half, 2>(iter, WelfordOps<at::Half, float, int32_t, float, std::pair<at::Half, at::Half>> { unbiased, take_sqrt }, WelfordData<float, int32_t, float> {});
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  sycl_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<out_t> (add_kernel<acc_t>()));
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  float factor = float(iter.num_output_elements()) / iter.numel();
  sycl_reduce_kernel<scalar_t, out_t>(iter, MeanOps<acc_t, float> {factor});
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
static void norm_kernel_impl(TensorIterator& iter, Scalar val) {
  float p;
  if (val.isIntegral()) {
     p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
     p = val.to<acc_t>();
  } else {
     AT_ERROR("norm_kernel_impl expects norm to be integer or float");
  }

  if (p == static_cast<float>(0)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<acc_t>(), 0);
  } else if (p == static_cast<float>(INFINITY)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<acc_t>(), std::numeric_limits<acc_t>::min());
  } else if (p == static_cast<float>(-INFINITY)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<acc_t>(), std::numeric_limits<acc_t>::max());
  } else {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{ acc_t(p) }, 0);
  }
}

static void std_var_kernel_sycl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std", [&]() {
    std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
  });
}

static void sum_kernel_sycl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "sum", [&]() {
    sum_kernel_impl<scalar_t>(iter);
  });
}

static void prod_kernel_sycl(TensorIterator& iter) {
  AT_ERROR("prod_kernel_sycl not implemented yet!");
}

static void mean_kernel_sycl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mean", [&]() {
    mean_kernel_impl<scalar_t>(iter);
  });
}

static void norm_kernel_sycl(TensorIterator& iter, Scalar p) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_impl<at::Half, float, float>(iter, p);
  }
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm", [&]() {
    norm_kernel_impl<scalar_t>(iter, p);
  });
}

void and_kernel_sycl(TensorIterator& iter) {
  sycl_reduce_kernel<uint8_t, uint8_t>(iter, func_wrapper<uint8_t> (and_kernel<uint8_t>()), true);
}

void or_kernel_sycl(TensorIterator& iter) {
  sycl_reduce_kernel<uint8_t, uint8_t>(iter, func_wrapper<uint8_t> (or_kernel<uint8_t>()), false);
}

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_sycl);
REGISTER_DISPATCH(sum_stub, &sum_kernel_sycl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_sycl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_sycl);
REGISTER_DISPATCH(and_stub, &and_kernel_sycl);
REGISTER_DISPATCH(or_stub, &or_kernel_sycl);
REGISTER_DISPATCH(norm_stub, &norm_kernel_sycl);
}
}
