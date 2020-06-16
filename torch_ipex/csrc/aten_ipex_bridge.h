#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include "cpu/dbl/Common.h"

#include <vector>

namespace torch_ipex {
namespace bridge {

// Convert DPCPP tensor to CPU tensor
at::Tensor shallowFallbackToCPUTensor(const at::Tensor& ipexTensor);
std::vector<at::Tensor> shallowFallbackToCPUTensorList(const at::TensorList&);

void attachShadeDataContext(const at::Tensor& tensor);

/**
 * Reorder the DNNL tensor to the public format if the input tensor contains DNNL tensor.
 *
 * @param[in] ipexTensor The DNNL tensor of the input ipex tensor to be reordered to public format
 */
void reorderDilTensorToPublic(const at::Tensor& ipexTensor);

/**
 * Reorder to a DNNL tensor with specified descriptor no matter input tensor is a DNNL tensor or not
 *
 * @param[in] ipexTensor The input tensor to be reordered to the spcified DNNL descriptor
 */
void reorderDilTensorGeneric(const at::Tensor& ipexTensor, const dil::tensor::desc& dstDesc);

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
