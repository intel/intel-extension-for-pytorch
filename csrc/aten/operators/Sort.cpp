
#include <ATen/AtenIpexTypeXPU.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "BitonicSort.h"
#include "MergeSort.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor&, Tensor&> sort_out(
    Tensor& sorted,
    Tensor& indices,
    const Tensor& input,
    long dim,
    bool order) {
  int64_t dims = sorted.dim() == 0 ? 1 : sorted.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  dims = input.dim() == 0 ? 1 : input.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  dims = indices.dim() == 0 ? 1 : indices.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  // Make sure sufficient output space is allocated
  sorted.resize_as_(input);
  indices.resize_(input.sizes());

  dim = maybe_wrap_dim(dim, input);

  // How large are the slices that we are sorting?
  int64_t sliceSize = input.dim() == 0 ? 1 : input.size(dim);
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int maxSliceSize = local_size * 2;
  if (sliceSize <= maxSliceSize) { // inplace sort
    // Fill `indices` (the values) with the
    // slice-relative index.
    at::AtenIpexTypeXPU::fill_slice_with_index(indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    sorted.copy_(input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        sorted.scalar_type(),
        "SortKeyValueInplace",
        [&]() { SortKeyValueInplace<scalar_t>(sorted, indices, dim, order); });
  } else {
    at::AtenIpexTypeXPU::fill_slice_with_index(indices, dim);
    int64_t item_size = sorted.size(dim);
    int64_t group_size = sorted.numel() / item_size;
    int64_t stride = sorted.stride(dim);
    sorted.copy_(input);
    std::vector<int64_t> scratch_size{sizeof(double) * sorted.numel()};
    Tensor scratch_key =
        at::empty(scratch_size, indices.options().dtype(at::ScalarType::Byte));
    Tensor scratch_value =
        at::empty(scratch_size, indices.options().dtype(at::ScalarType::Byte));
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        sorted.scalar_type(),
        "merge_sort_kernel",
        [&]() {
          merge_sort_kernel<scalar_t, int64_t>(
              sorted.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              group_size,
              item_size,
              order,
              (uint8_t*)scratch_key.data_ptr(),
              (uint8_t*)scratch_value.data_ptr(),
              stride);
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
