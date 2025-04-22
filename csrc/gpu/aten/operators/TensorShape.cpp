#include <ATen/ATen.h>

namespace at {
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

} // namespace at
