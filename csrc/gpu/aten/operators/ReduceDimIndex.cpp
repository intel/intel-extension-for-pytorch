#include <ATen/ATen.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/XPUPair.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct GreaterOrNan {
  bool operator()(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
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
  using arg_t = at::AtenIpexTypeXPU::pair<scalar_t, index_t>;

  static arg_t project(arg_t arg) {
    return arg;
  }

  static arg_t reduce(arg_t arg, scalar_t val, int64_t idx) {
    return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
  }

  static arg_t combine(arg_t a, arg_t b) {
    return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
  }

  static arg_t translate_idx(arg_t a, int64_t base_idx) {
    return {a.first, a.second + base_idx};
  }

#if 0
  static arg_t warp_shfl_down(arg_t arg, int offset) {
    return arg_t(
        WARP_SHFL_DOWN(arg.first, offset), WARP_SHFL_DOWN(arg.second, offset));
  }
#endif
};

template <typename scalar_t>
struct MaxOps : public MinMaxReductionOps<GreaterOrNan<scalar_t>> {};

std::tuple<Tensor&, Tensor&> _max_out(
    Tensor& max,
    Tensor& max_indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  dim = maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    at::native::zero_numel_tensor_resize(
        max, max_indices, self, dim, keepdim, "max()");
    return std::tie(max, max_indices);
  } else if (at::native::_dimreduce_return_trivial_no_ident(
                 max, self, dim, keepdim, "max")) {
    // case where self.numel() == 1. The result does not need to be reshaped
    // as a case of reduction in this case.
    TORCH_CHECK(!self.is_complex(), "max does not support complex inputs.");
    AT_ASSERT(max.dim() == 0);
    max_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(max, max_indices);
  } else {
    at::TensorIterator iter = make_reduction(
        "max", max, max_indices, self, dim, keepdim, self.scalar_type(), kLong);
    IPEX_DISPATCH_ALL_TYPES_AND3(
        kBFloat16, kHalf, kBool, iter.dtype(2), "max_xpu", [&]() {
          // register pressure is heavy when output vec size is large.
          // using 2 to mitigate register pressure to avoid register spill
          dpcpp_reduce_kernel<scalar_t, scalar_t, 4 /* vt0 */, 2 /* vt1 */>(
              iter,
              MaxOps<scalar_t>{},
              at::AtenIpexTypeXPU::pair<scalar_t, int64_t>(
                  Numerics<scalar_t>::lower_bound(), 0));
        });

    return {max, max_indices};
  }
}

std::tuple<Tensor&, Tensor&> max_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_values) {
  return AtenIpexTypeXPU::_max_out(max, max_values, self, dim, keepdim);
}

} // namespace AtenIpexTypeXPU
} // namespace at
