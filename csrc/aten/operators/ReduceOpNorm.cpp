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

// This accumulator template is used to calculate the order zero norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t>
struct NormOps {
  acc_t norm;

  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    return acc +
        Numerics<acc_t>::pow(
               static_cast<acc_t>(Numerics<scalar_t>::fabs(data)), norm);
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

  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }

  NormOps(acc_t norm) : norm(norm) {}
};

// This accumulator template is used to calculate the order zero norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t>
struct NormZeroOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    return acc + (data == static_cast<scalar_t>(0) ? acc_t(0) : acc_t(1));
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

// This accumulator template is used to calculate the order one norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t>
struct NormOneOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    return acc + static_cast<acc_t>(Numerics<scalar_t>::fabs(data));
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

// This accumulator template is used to calculate the maximum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t>
struct AbsMinOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    return Numerics<acc_t>::min(
        acc, static_cast<acc_t>(Numerics<scalar_t>::fabs(data)));
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

// This accumulator template is used to calculate the maximum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t>
struct AbsMaxOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    return Numerics<acc_t>::max(
        acc, static_cast<acc_t>(Numerics<scalar_t>::fabs(data)));
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <
    typename scalar_t,
    typename acc_t = typename scalar_value_type<scalar_t>::type,
    typename out_t = typename scalar_value_type<scalar_t>::type>
static void norm_kernel_impl(
    TensorIterator& iter,
    Scalar val,
    IntArrayRef dim) {
  float p = 0.0f;
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<acc_t>();
  } else {
    TORCH_CHECK(0, "norm_kernel_impl expects norm to be integer or float");
  }

  auto input = iter.tensor(iter.ntensors() - 1);
  if (p == static_cast<float>(0)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(
        iter, NormZeroOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(
        iter, NormOneOps<scalar_t, acc_t>(), 0);
  } else if (Numerics<float>::isinf(p)) {
    if (p < std::numeric_limits<float>::lowest()) {
      dpcpp_reduce_kernel<scalar_t, out_t>(
          iter,
          AbsMinOps<scalar_t, acc_t>(),
          std::numeric_limits<acc_t>::max());
    } else {
      dpcpp_reduce_kernel<scalar_t, out_t>(
          iter,
          AbsMaxOps<scalar_t, acc_t>(),
          std::numeric_limits<acc_t>::min());
    }
  } else {
    dpcpp_reduce_kernel<scalar_t, out_t>(
        iter, NormOps<scalar_t, acc_t>{acc_t(p)}, 0);
  }
  if (at::isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

static inline void norm_kernel(
    TensorIterator& iter,
    Scalar p,
    IntArrayRef dim) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_impl<at::Half, float>(iter, p, dim);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    return norm_kernel_impl<at::Half, float, float>(iter, p, dim);
  }
  if (iter.dtype() == kBFloat16) {
    return norm_kernel_impl<at::BFloat16, float>(iter, p, dim);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    return norm_kernel_impl<at::BFloat16, float, float>(iter, p, dim);
  }
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "norm",
      [&]() { norm_kernel_impl<scalar_t>(iter, p, dim); });
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
      "Can only calculate the mean of floating types or complex type. Got ",
      toString(scalarType),
      " instead.");

  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = meta::make_reduction("norm", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    norm_kernel(iter, p, dim);
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
        "norm only supports floating-point and complex dtypes");

    Tensor result;
    return at::AtenIpexTypeXPU::norm_out(
        result, self, p, IntArrayRef{}, false, c10::nullopt);
  }
}

// norm - customized outer entrances
Tensor& norm_out(
    const Tensor& self,
    const c10::optional<Scalar>& p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype,
    Tensor& out) {
  return at::AtenIpexTypeXPU::norm_out(
      out, self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor& norm_out(
    const Tensor& self,
    const c10::optional<Scalar>& p,
    IntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  return at::AtenIpexTypeXPU::norm_out(
      out, self, p, dim, keepdim, c10::nullopt);
}

Tensor& renorm_out(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm,
    Tensor& out) {
  TORCH_CHECK(!self.is_sparse(), "renorm(sycl_sparse) is not supported.");
  TORCH_CHECK(p.toFloat() > 0, "non-positive-norm not supported");
  TORCH_CHECK(self.dim() > 1, "need at least 2 dimensions, got ", self.dim());

  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());
  auto norm_vec_sz = self.size(dim);
  Tensor norm = at::empty(norm_vec_sz, self.options());
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

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "renorm_out",
      [&] {
        auto maxnorm_elm = maxnorm.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t norm) -> scalar_t {
          if (norm > maxnorm_elm)
            return maxnorm_elm / (norm + 1e-7);
          return 1;
        });
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

static Tensor& linalg_vector_norm_impl(
    const Tensor& self,
    const Scalar& scalar_ord,
    optional<IntArrayRef> opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  // Casting a large integer to a double will introduce some error, but for
  // practical purposes, it won't matter since a large order will usually
  // give an infinite result
  auto ord = scalar_ord.toDouble();

  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "linalg.vector_norm only supports strided layout, but got: ",
      self.layout());

  if (opt_dtype.has_value() && isComplexType(self.scalar_type())) {
    TORCH_CHECK(
        isComplexType(opt_dtype.value()),
        "linalg.vector_norm expected complex 'dtype', since input is complex, ",
        "but got ",
        opt_dtype.value());
  }

  ScalarType in_dtype = opt_dtype.value_or(self.scalar_type());
  TORCH_CHECK(
      at::isFloatingType(in_dtype) || at::isComplexType(in_dtype),
      "linalg.vector_norm only supports floating point and complex dtypes, but got: ",
      toString(in_dtype));

  IntArrayRef dim = opt_dim.value_or(IntArrayRef{});

  if (self.numel() == 0) {
    // TODO: The question about how to handle negative orders when the input
    // is empty has not been settled yet. For now, we raise an error. Issue:
    // https://github.com/pytorch/pytorch/issues/52783
    TORCH_CHECK(
        ord >= 0,
        "linalg.vector_norm of negative order cannot be performed on an empty tensor");

    // For NumPy compatibility, we can only perform order infinity reduction
    // (max/min) on a tensor with zero elements if the dimensions to reduce are
    // nonzero. Otherwise, throw an error.
    if (ord == INFINITY) {
      bool has_identity = true;

      if (dim.size() == 0) {
        has_identity = false;
      } else {
        for (int64_t dim_num : dim) {
          if (self.size(dim_num) == 0) {
            has_identity = false;
            break;
          }
        }
      }
      TORCH_CHECK(
          has_identity,
          "linalg.vector_norm cannot compute the infinity norm on an empty ",
          "dimension because the operation does not have an identity");
    }
  }
  Tensor self_;
  self_ = self;
  ScalarType out_dtype = opt_dtype.value_or(toValueType(self.scalar_type()));
  TORCH_CHECK(
      !result.defined() || out_dtype == result.scalar_type(),
      "linalg.vector_norm expected out tensor dtype ",
      out_dtype,
      " but got: ",
      result.scalar_type());
  // omit in_dtype in the following call, to avoid make_reduction explicitly
  // casting input to out_dtype
  auto iter = at::isComplexType(self.scalar_type())
      ? meta::make_reduction(
            "vector_norm", result, self_, dim, keepdim, in_dtype, out_dtype)
      : meta::make_reduction(
            "vector_norm", result, self_, dim, keepdim, out_dtype);

  norm_kernel(iter, ord, dim);
  return result;
}

Tensor linalg_vector_norm(
    const Tensor& self,
    const Scalar& ord,
    optional<IntArrayRef> opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  ScalarType out_dtype = opt_dtype.value_or(toValueType(self.scalar_type()));
  Tensor result = create_reduction_result(
      self, opt_dim.value_or(IntArrayRef{}), keepdim, out_dtype);
  return linalg_vector_norm_impl(
      self, ord, opt_dim, keepdim, opt_dtype, result);
}

Tensor& linalg_vector_norm_out(
    const Tensor& self,
    const Scalar& ord,
    optional<IntArrayRef> opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  return linalg_vector_norm_impl(
      self, ord, opt_dim, keepdim, opt_dtype, result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
