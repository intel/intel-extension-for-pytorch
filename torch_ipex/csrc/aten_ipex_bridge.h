#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

namespace torch_ipex {
namespace bridge {

at::Tensor fallbackToCPUTensor(const at::Tensor& ipexTensor);
at::Tensor upgradeToDPCPPTensor(const at::Tensor& ipexTensor);
void copyTensor(at::Tensor& dstTensor, const at::Tensor& scrTensor, c10::DeviceType devType);
at::TensorList fallbackToCPUTensorList(const at::TensorList&);
std::vector<at::Tensor> upgradeToDPCPPTensorVec(const std::vector<at::Tensor> &);

}  // namespace bridge
}  // namespace torch_ipex
