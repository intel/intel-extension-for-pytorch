#include <ATen/ATen.h>
#include <core/DeviceAllocator.h>
#include "comm/RegistrationDeclarations.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  recordStreamInDevAlloc(
      self.storage().data_ptr(),
      at::xpu::XPUStream::unpack3(
          stream.id(), stream.device_index(), stream.device_type()));
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  recordStreamInDevAlloc(
      self.storage().data_ptr(),
      at::xpu::XPUStream::unpack3(
          stream.id(), stream.device_index(), stream.device_type()));
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
