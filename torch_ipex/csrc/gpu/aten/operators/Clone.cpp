#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeXPU {

Tensor clone(const Tensor& self, c10::optional<MemoryFormat> memory_format) {
  return at::native::clone(self, memory_format);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor clone(const Tensor& self, c10::optional<MemoryFormat> memory_format) {
  return at::native::quantized_clone(self, memory_format);
}

} // namespace AtenIpexTypeXPU
} // namespace at
