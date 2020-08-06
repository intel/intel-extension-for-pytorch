#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor clone(const Tensor& self, c10::optional<MemoryFormat> memory_format) {
  if(self.is_quantized()){
	  return at::native::quantized_clone(self, memory_format);
  }
  return at::native::clone(self, memory_format);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
