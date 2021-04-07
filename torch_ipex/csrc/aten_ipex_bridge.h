#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/ATen.h>
#include <ATen/core/List.h>
#include "cpu/dbl/Common.h"

#include <vector>

namespace torch_ipex {
namespace bridge {

// Convert DPCPP tensor to CPU tensor
at::Tensor shallowFallbackToCPUTensor(const at::Tensor& ipexTensor);
std::vector<at::Tensor> shallowFallbackToCPUTensorList(const at::TensorList&);
c10::List<c10::optional<at::Tensor>> shallowFallbackToCPUTensorList(const c10::List<c10::optional<at::Tensor>>& tensor_list);
std::vector<at::Tensor> shallowFallbackToCPUTensorVec(const std::vector<at::Tensor> &tensor_vec);
void attachShadeDataContext(const at::Tensor& tensor);

// Convert CPU tensor to DPCPP tensor
at::Tensor shallowUpgradeToDPCPPTensor(const at::Tensor& ipexTensor);
std::vector<at::Tensor> shallowUpgradeToDPCPPTensorVec(const std::vector<at::Tensor> &);

// The last character A means alias. This function is for aten alias
//     ex: aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
at::Tensor shallowUpgradeToDPCPPTensorA(const at::Tensor& ipexTensor, const at::Tensor& cpuTensor);
// The last two character AW means alias and write. This function is for aten alias w/ write
//     ex: aten::asin_(Tensor(a!) self) -> Tensor(a!)
const at::Tensor& shallowUpgradeToDPCPPTensorAW(const at::Tensor& ipexTensor, const at::Tensor& cpuTensor);

}  // namespace bridge
}  // namespace torch_ipex
