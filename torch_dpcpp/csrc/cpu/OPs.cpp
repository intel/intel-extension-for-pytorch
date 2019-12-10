#include "cpu/OPs.h"

#include <ATen/Context.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/CPUGenerator.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>

#include "csrc/aten_ipex_bridge.h"

namespace torch_ipex {
namespace cpu {

at::Tensor AtenIpexCPUDefault::add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha) {
  auto _self = bridge::fallbackToCPUTensor(self);
  auto _other = bridge::fallbackToCPUTensor(other);
  auto res = at::add(_self, _other, alpha);
  return bridge::upgradeToDPCPPTensor(res);
}

at::Tensor AtenIpexCPUDefault::ones(at::IntArrayRef size, const at::TensorOptions & options) {
  at::TensorOptions o_options = options.device(at::DeviceType::CPU);
  auto ones_tensor = at::ones(size, o_options);
  return bridge::upgradeToDPCPPTensor(ones_tensor);
}

at::Tensor AtenIpexCPUDefault::empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format) {
  at::TensorOptions o_options = options.device(at::DeviceType::CPU);
  auto empty_tensor = at::empty(size, o_options, memory_format);
  return bridge::upgradeToDPCPPTensor(empty_tensor);
}

at::Tensor AtenIpexCPUDefault::view(const at::Tensor& self, at::IntArrayRef size) {
  TORCH_CHECK(false);
  return at::Tensor();
}

at::Tensor AtenIpexCPUDefault::to(const at::Tensor & self, c10::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(copy);
  TORCH_CHECK(device.type() == at::DeviceType::DPCPP);
  TORCH_CHECK(self.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(memory_format == at::MemoryFormat::Contiguous);
  TORCH_CHECK(self.scalar_type() == dtype);
  return bridge::upgradeToDPCPPTensor(self);
}

} // cpu
} // torch_ipex