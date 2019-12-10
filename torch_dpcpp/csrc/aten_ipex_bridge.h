#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

namespace torch_ipex {
namespace bridge {

at::Tensor fallbackToCPUTensor(const at::Tensor& ipexTensor);
at::Tensor upgradeToDPCPPTensor(const at::Tensor& ipexTensor);

}  // namespace bridge
}  // namespace torch_ipex
