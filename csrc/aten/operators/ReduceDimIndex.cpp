#include <ATen/ATen.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct LessOrNan {
  C10_DEVICE bool operator()(
      scalar_t a,
      scalar_t b,
      int64_t idx_a,
      int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else min(a, b)
    if (Numerics<scalar_t>::isnan(a)) {
      if (Numerics<scalar_t>::isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a < b);
  }
};

template <typename scalar_t>
struct GreaterOrNan {
  C10_DEVICE bool operator()(
      scalar_t a,
      scalar_t b,
      int64_t idx_a,
      int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else max(a, b)
    if (Numerics<scalar_t>::isnan(a)) {
      if (Numerics<scalar_t>::isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a > b);
  }
};

template <typename comp_t>
struct MinMaxReductionOps {
  using scalar_t = typename binary_function_traits<comp_t>::arg1_t;
  using index_t = int64_t;
  using arg_t = std::pair<scalar_t, index_t>;

  static C10_DEVICE arg_t project(arg_t arg) {
    return arg;
  }

  static C10_DEVICE arg_t reduce(arg_t arg, scalar_t val, int64_t idx) {
    return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
  }

  static C10_DEVICE arg_t combine(arg_t a, arg_t b) {
    return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
  }

  static C10_DEVICE arg_t translate_idx(arg_t a, int64_t base_idx) {
    return {a.first, a.second + base_idx};
  }

#if 0
  static C10_DEVICE arg_t warp_shfl_down(arg_t arg, int offset) {
    return arg_t(
        WARP_SHFL_DOWN(arg.first, offset), WARP_SHFL_DOWN(arg.second, offset));
  }
#endif
};

template <typename scalar_t>
struct MinOps : public MinMaxReductionOps<LessOrNan<scalar_t>> {};

template <typename scalar_t>
struct MaxOps : public MinMaxReductionOps<GreaterOrNan<scalar_t>> {};

std::tuple<Tensor&, Tensor&> _min_out(
    Tensor& min,
    Tensor& min_indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  at::TensorIterator iter = make_reduction(
      "min", min, min_indices, self, dim, keepdim, self.scalar_type(), kLong);
  IPEX_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(2), "min_xpu", [&]() {
        dpcpp_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MinOps<scalar_t>{},
            std::pair<scalar_t, int64_t>(Numerics<scalar_t>::upper_bound(), 0));
      });

  return {min, min_indices};
}

std::tuple<Tensor, Tensor> _min(const Tensor& self, int64_t dim, bool keepdim) {
  auto min = at::empty({0}, self.options());
  auto min_indices = at::empty({0}, self.options().dtype(kLong));
  return AtenIpexTypeXPU::_min_out(min, min_indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim) {
  return AtenIpexTypeXPU::_min(self, dim, keepdim);
}

std::tuple<Tensor&, Tensor&> min_out(
    Tensor& min,
    Tensor& min_values,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return AtenIpexTypeXPU::_min_out(min, min_values, self, dim, keepdim);
}

std::tuple<Tensor&, Tensor&> _max_out(
    Tensor& max,
    Tensor& max_indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  at::TensorIterator iter = make_reduction(
      "max", max, max_indices, self, dim, keepdim, self.scalar_type(), kLong);
  IPEX_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(2), "max_xpu", [&]() {
        dpcpp_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MaxOps<scalar_t>{},
            std::pair<scalar_t, int64_t>(Numerics<scalar_t>::lower_bound(), 0));
      });

  return {max, max_indices};
}

std::tuple<Tensor, Tensor> _max(const Tensor& self, int64_t dim, bool keepdim) {
  auto max = at::empty({0}, self.options());
  auto max_indices = at::empty({0}, self.options().dtype(kLong));
  return AtenIpexTypeXPU::_max_out(max, max_indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim) {
  return AtenIpexTypeXPU::_max(self, dim, keepdim);
}

std::tuple<Tensor&, Tensor&> max_out(
    Tensor& max,
    Tensor& max_values,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return AtenIpexTypeXPU::_max_out(max, max_values, self, dim, keepdim);
}

} // namespace AtenIpexTypeXPU
} // namespace at
