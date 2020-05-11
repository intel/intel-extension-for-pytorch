#pragma once

#include <ATen/Tensor.h>
#include "cpu/dil/dil.hpp"

namespace torch_ipex {

enum DPCPPSubDev {
    CPU,
};

bool check_device_by_tensor(const at::Tensor& tensor, DPCPPSubDev sub_dev);
bool check_device_by_tensor_list(const at::TensorList& tensor_list, DPCPPSubDev sub_dev);
bool check_device_by_options(const at::TensorOptions& tensor_options, DPCPPSubDev sub_dev);
bool check_device_by_device(const at::Device& device, DPCPPSubDev sub_dev);
bool check_layout_by_options(const at::TensorOptions& tensor_options, c10::Layout layout);
bool get_device_count(c10::Device dev_type, c10::DeviceIndex *count);
dil::data_type get_dil_data_type(at::ScalarType);
at::ScalarType get_at_data_type(dil::data_type);
bool check_auto_dnnl();
bool check_tensor_own_whole_storage(const at::Tensor& tensor);
bool check_tensor_own_shade_context(const at::Tensor& tensor);

} // namespace torch_ipex
