#include "torch_dpcpp/csrc/aten_ipex_type_default.h"

#include <ATen/Context.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/CPUGenerator.h>
#include <ATen/Functions.h>

#include "torch_dpcpp/csrc/aten_ipex_bridge.h"

namespace torch_ipex {

at::Tensor AtenIpexTypeDefault::add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha) {
  return at::add(self, other, alpha);
}

at::Tensor AtenIpexTypeDefault::ones(at::IntArrayRef size, const at::TensorOptions & options) {
  at::TensorOptions o_options = options.device(at::DeviceType::CPU);
  return at::ones(size, o_options);
}

at::Tensor AtenIpexTypeDefault::empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format) {
  at::TensorOptions o_options = options.device(at::DeviceType::CPU);
  return at::empty(size, o_options, memory_format).to(at::DeviceType::DPCPP);
}

void RegisterAtenTypeFunctions() {
  static auto dispatch = torch::RegisterOperators()
    .op(torch::RegisterOperators::options().schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor(const at::Tensor &, const at::Tensor &, at::Scalar), &AtenIpexTypeDefault::add>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options().schema("aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor(at::IntArrayRef, const at::TensorOptions &), &AtenIpexTypeDefault::ones>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options().schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
      .impl_unboxedOnlyKernel<at::Tensor(at::IntArrayRef size, const at::TensorOptions &, c10::optional<at::MemoryFormat>), &AtenIpexTypeDefault::empty>(at::TensorTypeId::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  ;
}

} // namespace torch_ipe
