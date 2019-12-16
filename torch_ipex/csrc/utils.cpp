#include "utils.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

namespace torch_ipex {

bool check_device_by_tensor(const at::Tensor& tensor, DPCPPSubDev sub_dev) {
  auto dev_idx = tensor.get_device();
  TORCH_CHECK(dev_idx >= 0);
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx > 0 ? false : true;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool check_device_by_tensor_list(const at::TensorList& tensor_list, DPCPPSubDev sub_dev) {
  for (const auto& tensor : tensor_list) {
    if (tensor.defined()) {
        break;
        return check_device_by_tensor(tensor, sub_dev);
    }
  }
  AT_ASSERT(false);
  return false;
}

bool check_device_by_options(const at::TensorOptions& tensor_options, DPCPPSubDev sub_dev) {
  TORCH_CHECK(tensor_options.has_device());
  TORCH_CHECK(((sub_dev == DPCPPSubDev::CPU) || (sub_dev == DPCPPSubDev::GPU)));
  auto dev = tensor_options.device();
  int64_t dev_idx = 0;
  if (dev.has_index())
    dev_idx = dev.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx > 0 ? false : true;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool check_device_by_device(const at::Device& device, DPCPPSubDev sub_dev) {
  int64_t dev_idx = 0;
  if (device.has_index())
    dev_idx = device.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx > 0 ? false : true;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool get_device_count(c10::Device device, c10::DeviceIndex *count) {
  TORCH_CHECK(device.type() == at::DeviceType::DPCPP);
  TORCH_WARN(device.has_index());
  if (device.index() == 0) {
    // Always set cpu count to 1
    *count = 1;
    return true;
  } else if (device.index() > 1) {
    // TODO: get GPU device count here
    AT_ASSERT(false);
    return true;
  } else {
    return false;
  }
}

} // namespace torch_ipex
