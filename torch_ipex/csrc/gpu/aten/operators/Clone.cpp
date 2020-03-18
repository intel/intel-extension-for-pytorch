#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor clone(const Tensor &self, c10::optional<MemoryFormat> memory_format) {
  return at::native::clone(self, memory_format);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
