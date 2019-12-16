#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {

enum DPCPPSubDev {
    CPU,
    GPU
};

bool check_device_by_tensor(const at::Tensor& tensor, DPCPPSubDev sub_dev);
bool check_device_by_tensor_list(const at::TensorList& tensor_list, DPCPPSubDev sub_dev);
bool check_device_by_options(const at::TensorOptions& tensor_options, DPCPPSubDev sub_dev);
bool check_device_by_device(const at::Device& device, DPCPPSubDev sub_dev);
bool get_device_count(c10::Device dev_type, c10::DeviceIndex *count);

} // namespace torch_ipex
