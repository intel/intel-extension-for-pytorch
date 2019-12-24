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

// Copy tensor raw data
void copyTensor(at::Tensor& dstTensor, const at::Tensor& scrTensor, c10::DeviceType devType);

// Convert number of DPCPP tensors to CPU tensor
at::TensorList fallbackToCPUTensorList(const at::TensorList&);
at::TensorList shallowFallbackToCPUTensorList(const at::TensorList&);

// Convert number of CPU tensors to DPCPP tensor
std::vector<at::Tensor> upgradeToDPCPPTensorVec(const std::vector<at::Tensor> &);
std::vector<at::Tensor> shallowUpgradeToDPCPPTensorVec(const std::vector<at::Tensor> &);
}  // namespace bridge
}  // namespace torch_ipex
