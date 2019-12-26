#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

namespace torch_ipex {
namespace bridge {

// Convert DPCPP tensor to CPU tensor
at::Tensor fallbackToCPUTensor(const at::Tensor& ipexTensor);
at::Tensor shallowFallbackToCPUTensor(const at::Tensor& ipexTensor);

// Convert CPU tensor to DPCPP tensor
at::Tensor upgradeToDPCPPTensor(const at::Tensor& ipexTensor);
at::Tensor shallowUpgradeToDPCPPTensor(const at::Tensor& ipexTensor);

// The last character A means alias. This function is for aten alias
//     ex: aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
at::Tensor shallowUpgradeToDPCPPTensorA(const at::Tensor& ipexTensor, const at::Tensor& cpuTensor);
// The last two character AW means alias and write. This function is for aten alias w/ write
//     ex: aten::asin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor& shallowUpgradeToDPCPPTensorAW(at::Tensor& ipexTensor, at::Tensor& cpuTensor);
at::Tensor& shallowUpgradeToDPCPPTensorInplace(at::Tensor& ipexTensor, at::Tensor& cpuTensor);

// Copy tensor raw data
void copyTensor(at::Tensor& dstTensor, const at::Tensor& scrTensor, c10::DeviceType devType);

// Convert number of DPCPP tensors to CPU tensor
std::vector<at::Tensor> fallbackToCPUTensorList(const at::TensorList&);
std::vector<at::Tensor> shallowFallbackToCPUTensorList(const at::TensorList&);

// Convert number of CPU tensors to DPCPP tensor
std::vector<at::Tensor> upgradeToDPCPPTensorVec(const std::vector<at::Tensor> &);
std::vector<at::Tensor> shallowUpgradeToDPCPPTensorVec(const std::vector<at::Tensor> &);
}  // namespace bridge
}  // namespace torch_ipex
