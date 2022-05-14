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

template <typename acc_t>
struct NormOps {
  acc_t norm;

  inline acc_t reduce(acc_t acc, acc_t data, int64_t idx) const {
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

  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }

  NormOps(acc_t norm) : norm(norm) {}
};

template <typename acc_t>
struct NormZeroOps {
  inline acc_t reduce(acc_t acc, acc_t data, int64_t idx) const {
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

template <typename acc_t>
struct NormOneOps {
  inline acc_t reduce(acc_t acc, acc_t data, int64_t idx) const {
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

template <typename acc_t>
struct AbsMinOps {
  inline acc_t reduce(acc_t acc, acc_t data, int64_t idx) const {
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

template <typename acc_t>
struct AbsMaxOps {
  inline acc_t reduce(acc_t acc, acc_t data, int64_t idx) const {
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
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
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
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<acc_t>(), 0);
  } else if (Numerics<float>::isinf(p)) {
    if (p < std::numeric_limits<float>::lowest()) {
      dpcpp_reduce_kernel<scalar_t, out_t>(
          iter, AbsMinOps<acc_t>(), std::numeric_limits<acc_t>::max());
    } else {
      dpcpp_reduce_kernel<scalar_t, out_t>(
          iter, AbsMaxOps<acc_t>(), std::numeric_limits<acc_t>::min());
    }
  } else {
    dpcpp_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{acc_t(p)}, 0);
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
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
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
      "Can only calculate the mean of floating types. Got ",
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
        "norm only supports floating-point dtypes");

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
  dpcpp_kernel_for_tensor_iter(iter, [=](float norm) -> float {
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

} // namespace AtenIpexTypeXPU
} // namespace at
