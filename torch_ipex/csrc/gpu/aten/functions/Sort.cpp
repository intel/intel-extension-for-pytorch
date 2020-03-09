#include <core/SYCLMemory.h>
#include <core/SYCL.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/Numerics.h>
#include <ATen/aten_ipex_type_dpcpp.h>

#include "Sort.h"


using namespace at::sycl::detail;

namespace at {
namespace AtenIpexTypeDPCPP {

std::tuple<Tensor &, Tensor &>
sort_out(Tensor & sorted, Tensor & indices, const Tensor & input, long dim, bool order) {
  int64_t dims = sorted.dim() == 0 ? 1 : sorted.dim();
  TORCH_CHECK(dims <= MAX_SYCLTORCH_DIMS, SYCLTORCH_DIM_WARNING);
  dims = input.dim() == 0 ? 1 : input.dim();
  TORCH_CHECK(dims <= MAX_SYCLTORCH_DIMS, SYCLTORCH_DIM_WARNING);
  dims = indices.dim() == 0 ? 1 : indices.dim();
  TORCH_CHECK(dims <= MAX_SYCLTORCH_DIMS, SYCLTORCH_DIM_WARNING);

  // Make sure sufficient output space is allocated
  sorted.resize_as_(input);
  indices.resize_(input.sizes());

  dim = maybe_wrap_dim(dim, input);

  // How large are the slices that we are sorting?
  int64_t sliceSize = input.dim() == 0 ? 1 : input.size(dim);
  int maxSliceSize = 2048;
  if (sliceSize <= maxSliceSize) { // inplace sort
    // Fill `indices` (the values) with the
    // slice-relative index.
    at::AtenIpexTypeDPCPP::fill_slice_with_index(indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    sorted.copy_(input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, sorted.scalar_type(), "SortKeyValueInplace",
        [&] () {
          SortKeyValueInplace<scalar_t>(sorted, indices, dim, order);
        }
    );
  } else {
    // TODO
  }

  return std::tuple<Tensor &, Tensor &>(sorted, indices);
}

std::tuple<at::Tensor,at::Tensor>
sort(const at::Tensor & self, int64_t dim, bool descending) {
  auto sorted = at::empty_like(self);
  auto indices = at::empty({0}, self.options().dtype(kLong));
  return at::AtenIpexTypeDPCPP::sort_out(sorted, indices, self, dim, descending);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
