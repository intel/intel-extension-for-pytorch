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

Tensor narrow_copy(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  return at::native::narrow_copy_dense(self, dim, start, length);
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

const Tensor& as_strided_(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<int64_t> storage_offset_) {
  return at::native::as_strided_(self, size, stride, storage_offset_);
}

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::transpose_(self, dim0, dim1);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
