#include <ATen/ATen.h>
#include <core/CachingAllocator.h>

namespace at {
namespace AtenIpexTypeXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  at::dpcpp::dpcpp_recordQueue(self.storage().data_ptr(), at::dpcpp::DPCPPStream::unpack(stream.pack()));
}
}  // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  at::dpcpp::dpcpp_recordQueue(self.storage().data_ptr(), at::dpcpp::DPCPPStream::unpack(stream.pack()));
}
}  // namespace AtenIpexTypeQuantizedXPU
}  // namespace at
