#include "aten_ipex_type_default.h"

#include <ATen/Context.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/CPUGenerator.h>
#include <ATen/Functions.h>

#include <c10/util/Exception.h>

#include "aten_ipex_bridge.h"
#include "utils.h"
#include "cpu/OPs.h"

namespace torch_ipex {

at::Tensor AtenIpexTypeDefault::add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha) {
  if (check_device(self, DPCPPSubDev::CPU)) {
    return cpu::AtenIpexCPUDefault::add(self, other, alpha);
  } else {
    AT_ASSERT(false);
  }
}

at::Tensor AtenIpexTypeDefault::ones(at::IntArrayRef size, const at::TensorOptions & options) {
  if (check_device(options, DPCPPSubDev::CPU)) {
    return cpu::AtenIpexCPUDefault::ones(size, options);
  } else {
    AT_ASSERT(false);
  }
}

at::Tensor AtenIpexTypeDefault::empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format) {
  at::TensorOptions o_options = options.device(at::DeviceType::CPU);
  auto empty_tensor = at::empty(size, o_options, memory_format);
  return bridge::upgradeToDPCPPTensor(empty_tensor);
}

at::Tensor AtenIpexTypeDefault::view(const at::Tensor& self, at::IntArrayRef size) {
  assert(false);
  return at::Tensor();
}

at::Tensor AtenIpexTypeDefault::to(const at::Tensor & self, c10::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(copy);
  TORCH_CHECK(device.type() == at::DeviceType::DPCPP);
  TORCH_CHECK(self.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(memory_format == at::MemoryFormat::Contiguous);
  TORCH_CHECK(self.scalar_type() == dtype);
  return bridge::upgradeToDPCPPTensor(self);
}

void RegisterAtenTypeFunctions() {
  static auto dispatch = torch::RegisterOperators()
    .op(torch::RegisterOperators::options().schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor(const at::Tensor &, const at::Tensor &, at::Scalar), &AtenIpexTypeDefault::add>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options().schema("aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor(at::IntArrayRef, const at::TensorOptions &), &AtenIpexTypeDefault::ones>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options()
      .schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor (at::IntArrayRef, const at::TensorOptions &, c10::optional<at::MemoryFormat>), &AtenIpexTypeDefault::empty>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options()
      .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
      .impl_unboxedOnlyKernel<at::Tensor(const at::Tensor &, at::IntArrayRef), &AtenIpexTypeDefault::view>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options()
      .schema("aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor(const at::Tensor &, c10::Device, at::ScalarType, bool, bool, c10::optional<at::MemoryFormat>), &AtenIpexTypeDefault::to>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  ;
}

} // namespace torch_ipe
