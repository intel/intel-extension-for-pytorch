#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>

#include <assert.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "SortingFastGroupSelect.h"
#include "SortingFastGroupSort.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void topk_out_with_sort(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    const Tensor& values,
    const Tensor& indices) {
  Tensor sorted_values, sorted_indices;
  std::tie(sorted_values, sorted_indices) =
      at::AtenIpexTypeXPU::sort(self, /* stable= */ false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

bool should_use_sort(const Tensor& self, int64_t dim, int64_t k) {
  // Affected by L1 cache capacity, too large k value cannot be used for fast
  // path.
  // TODO: Use radix sort for large problem size.
  return k > 256;
}

std::tuple<at::Tensor&, at::Tensor&> topk_out(
    const at::Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices) {
  if (k == 0) {
    return std::forward_as_tuple(values, indices);
  }

  TORCH_CHECK(
      input.defined() && values.defined() && indices.defined(),
      "invalid inputs");

  auto self = (input.dim() == 0) ? input.view(1) : input;

  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nelements = self.sizes()[dim];

  if (should_use_sort(self, dim, k)) {
    topk_out_with_sort(self, k, dim, largest, values, indices);
    return std::forward_as_tuple(values, indices);
  }

  TORCH_CHECK(
      nelements <= std::numeric_limits<int>::max(),
      "The dimension being select can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(
      self_dtype != ScalarType::ComplexFloat &&
          self_dtype != ScalarType::ComplexDouble,
      "Topk currently does not support complex dtypes on XPU.");

  auto out_sizes = self.sizes().vec();
  out_sizes[dim] = k;
  values.resize_(out_sizes);
  indices.resize_(out_sizes);

  Tensor self_;
  bool need_infer_dim = dim != ndim - 1;
  if (!need_infer_dim) {
    self_ = self.contiguous();
  } else {
    self_ = self.transpose(ndim - 1, dim).contiguous();
    std::swap(out_sizes[ndim - 1], out_sizes[dim]);
  }

  Tensor values_, indices_;
  bool newvalues = false;
  bool newindices = false;
  if (!need_infer_dim && values.is_contiguous()) {
    values_ = values;
  } else {
    values_ = at::empty(out_sizes, values.options());
    newvalues = true;
  }
  if (!need_infer_dim && indices.is_contiguous()) {
    indices_ = indices;
  } else {
    indices_ = at::empty(out_sizes, indices.options());
    newindices = true;
  }

  int64_t nsegments = numel / nelements;

  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self_.scalar_type(),
      "topk_out",
      [&]() {
        scalar_t* self_ptr = self_.data_ptr<scalar_t>();
        scalar_t* values_ptr = values_.data_ptr<scalar_t>();
        int64_t* indices_ptr = indices_.data_ptr<int64_t>();
        fast_group_select_pairs<scalar_t, int64_t>(
            self_ptr,
            (scalar_t*)values_ptr,
            nullptr,
            (int64_t*)indices_ptr,
            nsegments,
            nelements,
            k,
            largest);

        if (sorted) {
          fast_group_sort_pairs<scalar_t, int64_t, int64_t, false>(
              values_ptr,
              values_ptr,
              indices_ptr,
              indices_ptr,
              nsegments,
              k,
              largest);
        }
      });

  if (newvalues) {
    if (need_infer_dim)
      values_.transpose_(ndim - 1, dim);
    values.copy_(values_);
  }
  if (newindices) {
    if (need_infer_dim)
      indices_.transpose_(ndim - 1, dim);
    indices.copy_(indices_);
  }

  return std::forward_as_tuple(values, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
