#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor&, Tensor&> _min_out(
    Tensor& min,
    Tensor& min_indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      min.scalar_type(),
      "_min_out",
      [&] {
        std::pair<scalar_t, int64_t> init = std::make_pair<scalar_t, int64_t>(
            Numerics<scalar_t>::upper_bound(), 0);

        reduceDimIndex<scalar_t, int64_t>(
            min,
            min_indices,
            self,
            dim,
            keepdim,
            init,
            MinValuePair<scalar_t, int64_t>());
      });

  return std::tuple<Tensor&, Tensor&>(min, min_indices);
}

std::tuple<Tensor, Tensor> _min(const Tensor& self, int64_t dim, bool keepdim) {
  auto min = empty({0}, self.options());
  auto min_indices = empty({0}, self.options().dtype(kLong));

  dim = maybe_wrap_dim(dim, TensorImpl_Unwrap(self));

  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min")) {
    AT_ASSERT(min.dim() == 0);
    min_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(min, min_indices);
  } else {
    return AtenIpexTypeXPU::_min_out(min, min_indices, self, dim, keepdim);
  }
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
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      max.scalar_type(),
      "_max_out",
      [&] {
        std::pair<scalar_t, int64_t> init = std::make_pair<scalar_t, int64_t>(
            Numerics<scalar_t>::lower_bound(), 0);

        reduceDimIndex<scalar_t, int64_t>(
            max,
            max_indices,
            self,
            dim,
            keepdim,
            init,
            MaxValuePair<scalar_t, int64_t>());
      });

  return std::tuple<Tensor&, Tensor&>(max, max_indices);
}

std::tuple<Tensor, Tensor> _max(const Tensor& self, int64_t dim, bool keepdim) {
  auto max = empty({0}, self.options());
  auto max_indices = empty({0}, self.options().dtype(kLong));

  dim = maybe_wrap_dim(dim, TensorImpl_Unwrap(self));

  if (_dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    AT_ASSERT(max.dim() == 0);
    max_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(max, max_indices);
  } else {
    return AtenIpexTypeXPU::_max_out(max, max_indices, self, dim, keepdim);
  }
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
