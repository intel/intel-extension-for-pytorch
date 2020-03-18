#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor as_strided(
    const Tensor& self,
    IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
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

} // AtenIpexTypeDPCPP
} // at
