#include <ATen/ATen.h>
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor as_strided(
    const Tensor& self,
    IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  if (self.is_quantized()) {
    return at::native::as_strided_qtensorimpl(
        self, size, stride, storage_offset);
  }
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

Tensor view(const Tensor& self, IntArrayRef size) {
  return at::native::view(self, size);
}

Tensor unfold(
    const Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

// NOTE [ Unsafe View ]
// _unsafe_view() differs from view() in that the returned tensor isn't treated
// as a view for the purposes of automatic differentiation. (It's not listed in
// VIEW_FUNCTIONS in gen_inplace_or_view_type.py).  It's only safe to use if the
// `self` tensor is temporary. For example, the viewed tensor here (a + b) is
// discarded immediately after viewing:
//
//  res = at::_unsafe_view(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.
Tensor _unsafe_view(const Tensor& self, IntArrayRef size) {
  return self.view(size);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor view(const Tensor& self, IntArrayRef size) {
  return at::native::view(self, size);
}

Tensor as_strided(
    const Tensor& self,
    IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  return at::native::as_strided_qtensorimpl(self, size, stride, storage_offset);
}

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::transpose_(self, dim0, dim1);
}
} // namespace AtenIpexTypeQuantizedXPU

namespace AtenIpexTypeSparseXPU {

Tensor narrow_copy(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  int64_t allDim = self.dim();
  int64_t end = start + length;
  TORCH_CHECK(allDim > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(length >= 0, "narrow(): length must be non-negative.");
  TORCH_CHECK(
      dim >= 0 && dim < allDim,
      "Dimension ",
      dim,
      " out of range. Expecting 0 <= dim < ",
      allDim,
      ".");
  TORCH_CHECK(
      start >= 0 && end <= self.size(dim),
      "Invalid range to narrow. range(start, start+length) must be a subset of range(0, ",
      self.size(dim),
      ").")
  Tensor indices = self._indices();
  int64_t sparse_dim = self.sparse_dim();

  std::vector<int64_t> new_sizes = self.sizes().vec();
  new_sizes[dim] = length;

  Tensor new_values;
  Tensor new_indices;
  if (dim < sparse_dim) {
    Tensor mask = (indices[dim] >= start).__and__((indices[dim] < end));
    new_indices = indices.masked_select(mask).view({sparse_dim, -1});
    new_indices[dim].sub_(start);
    Tensor nzIndices = mask.nonzero().view(-1);
    new_values = self._values().index_select(0, nzIndices);
  } else {
    /* This means we are narrowing on a dense dim, which is in effect just a
        regular narrow on _values() */
    new_indices = indices;
    int64_t dense_dim = dim - sparse_dim + 1;
    new_values = self._values().narrow_copy(dense_dim, start, length);
  }

  auto newTensor = at::sparse_coo_tensor(new_indices, new_values, new_sizes);
  return newTensor._coalesced_(self.is_coalesced());
}
} // namespace AtenIpexTypeSparseXPU
} // namespace at
