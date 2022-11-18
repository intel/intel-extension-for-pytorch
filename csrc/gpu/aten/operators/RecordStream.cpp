#include <ATen/ATen.h>
#include <core/Allocator.h>
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  recordStreamInDevAlloc(
      self.storage().data_ptr(), DPCPPStream::unpack(stream.pack()));
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  recordStreamInDevAlloc(
      self.storage().data_ptr(), DPCPPStream::unpack(stream.pack()));
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
