#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "BitonicMergeSort.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& fill_slice_with_index(at::Tensor& t, int dim);

std::tuple<Tensor&, Tensor&> sort_out(
    Tensor& sorted,
    Tensor& indices,
    const Tensor& input,
    long dim,
    bool order) {
  if (input.numel() == 0)
    return {sorted, indices};

  int64_t dims = sorted.dim() == 0 ? 1 : sorted.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  dims = input.dim() == 0 ? 1 : input.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  dims = indices.dim() == 0 ? 1 : indices.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  // Make sure sufficient output space is allocated
  sorted.resize_as_(input);
  indices.resize_(input.sizes());

  if (input.dim() == 1 && input.numel() == 0) {
    return std::tuple<Tensor&, Tensor&>(sorted, indices);
  }

  // check if input is scalar
  if (input.dim() == 0 && input.numel() == 1) {
    indices.zero_();
    sorted.copy_(input);
    return std::tuple<Tensor&, Tensor&>(sorted, indices);
  }

  dim = maybe_wrap_dim(dim, input);

  // How large are the slices that we are sorting?
  int64_t sliceSize = input.dim() == 0 ? 1 : input.size(dim);

  at::AtenIpexTypeXPU::fill_slice_with_index(indices, dim);

  int64_t prb_size = sorted.size(dim);
  int64_t stride = sorted.stride(dim);
  int64_t batch_size = sorted.numel() / prb_size / stride;

  sorted.copy_(input);

  if (!order) {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        sorted.scalar_type(),
        "bitonic_merge_sort_kernel",
        [&]() {
          bitonic_merge_sort_kernel<scalar_t, int64_t>(
              sorted.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              prb_size,
              batch_size,
              stride,
              Numerics<scalar_t>::upper_bound(),
              [](scalar_t a, scalar_t b) -> bool {
                return Numerics<scalar_t>::lt(a, b);
              });
        });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        sorted.scalar_type(),
        "bitonic_merge_sort_kernel",
        [&]() {
          bitonic_merge_sort_kernel<scalar_t, int64_t>(
              sorted.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              prb_size,
              batch_size,
              stride,
              Numerics<scalar_t>::lower_bound(),
              [](scalar_t a, scalar_t b) -> bool {
                return Numerics<scalar_t>::gt(a, b);
              });
        });
  }

  return std::tuple<Tensor&, Tensor&>(sorted, indices);
}

std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  auto sorted = at::empty_like(self);
  auto indices = at::empty({0}, self.options().dtype(kLong));
  return at::AtenIpexTypeXPU::sort_out(sorted, indices, self, dim, descending);
}

} // namespace AtenIpexTypeXPU
} // namespace at
