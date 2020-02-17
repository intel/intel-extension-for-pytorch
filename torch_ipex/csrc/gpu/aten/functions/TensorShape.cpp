#include <ATen/NativeFunctions.h>

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor as_strided(const at::Tensor & self, at::IntArrayRef size,
    at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

at::Tensor view(const at::Tensor & self, at::IntArrayRef size) {
  return at::native::view(self, size);
}

} // AtenIpexTypeDPCPP
} // at
